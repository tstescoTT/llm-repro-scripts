#!/usr/bin/env python3
"""
Batch Local LLM Inference Script v3
Processes batch requests to a locally loaded LLM model using the EXACT same inference approach as text_demo.py.
The only difference from text_demo.py is the prompts being sent.

CLI Options:
  --batch_size: Number of users in a batch (default: 1, supports 1/2/4/8/16/32)
  --repeat_batches: Number of times to repeat the batch processing (default: 3)
  --max_seq_len: Maximum context length supported by the model (default: 128 * 1024)
  --max_generated_tokens: Maximum number of tokens to generate (default: 128)
  --prompt_file: Path to JSON file containing the prompt (default: sample_prompts/repro_prompt_ISL_3175.json)
"""

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from datetime import datetime
import os
import argparse
from pathlib import Path
import re
import torch
import ttnn
from loguru import logger

# Import everything from text_demo.py
from text_demo import (
    create_tt_model,
    LlamaOptimizations,
    SamplingParams,
    preprocess_inputs_prefill,
    Tokenizer,
    Generator,
    PagedAttentionConfig,
)

# Generate timestamped filename at startup
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = "output"
JSON_FILENAME = os.path.join(OUTPUT_DIR, f"batch_local_inference_{TIMESTAMP}.json")

# Global variable to track if JSON file has been initialized
json_file_initialized = False


def write_batch_results_incremental(batch_results, metadata=None, final_metadata=None, is_final=False):
    """Write batch results to JSON file incrementally after each batch completion"""
    global json_file_initialized
    
    try:
        if not json_file_initialized:
            # Initialize JSON file with metadata and start results array
            with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
                f.write('{\n')
                if metadata:
                    f.write(f'  "metadata": {json.dumps(metadata, ensure_ascii=False, indent=2)},\n')
                f.write('  "results": [\n')
                
                # Write first batch results
                for i, result in enumerate(batch_results):
                    if i > 0:
                        f.write(',\n')
                    f.write('    ' + json.dumps(result, ensure_ascii=False, indent=4).replace('\n', '\n    '))
                
                f.flush()
            
            json_file_initialized = True
            logger.info(f"Initialized JSON file and wrote {len(batch_results)} results to {JSON_FILENAME}")
        
        else:
            # Append new batch results
            with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(',\n')
                    f.write('    ' + json.dumps(result, ensure_ascii=False, indent=4).replace('\n', '\n    '))
                f.flush()
            
            logger.info(f"Appended {len(batch_results)} results to {JSON_FILENAME}")
        
        # Close the JSON file if this is the final write
        if is_final:
            with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                f.write('\n  ]')
                
                # Add final metadata if provided
                if final_metadata:
                    f.write(',\n')
                    f.write(f'  "final_metadata": {json.dumps(final_metadata, ensure_ascii=False, indent=2)}')
                
                f.write('\n}\n')
                f.flush()
            
            logger.info(f"Finalized JSON file: {JSON_FILENAME}")
                
    except Exception as e:
        logger.error(f"Error writing batch results to JSON: {e}")
        raise


def load_prompt_from_file(prompt_file_path):
    """Load prompt from JSON file"""
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(prompt_data, list) and len(prompt_data) > 0:
            # Array format like repro_prompt_ISL_3175.json
            return prompt_data[0].get("prompt", "")
        elif isinstance(prompt_data, dict):
            # Direct object format
            return prompt_data.get("prompt", "")
        else:
            logger.error(f"Unexpected JSON structure in {prompt_file_path}")
            return ""
            
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_file_path}: {e}")
        return ""


# Default request payload structure (will be populated from JSON file)
REQUEST_PAYLOAD = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "temperature": 0.0,
    "max_tokens": 1000,
    "stream": False,
    "messages": []
}


def create_model_symlink(symlinks_dir, model_name, weights_dir, file_symlinks_map={}):
    """Helper function to create and manage model symlinks."""
    symlink_path = symlinks_dir / model_name

    # Handle file-specific symlinks (for vision models)
    if file_symlinks_map:
        # Clean up any existing symlinks
        if symlink_path.exists():
            for _link in symlink_path.iterdir():
                if _link.is_symlink():
                    _link.unlink()
        symlink_path.mkdir(parents=True, exist_ok=True)

        # Create individual file symlinks
        for target_file, source_file in file_symlinks_map.items():
            (symlink_path / target_file).symlink_to(weights_dir / source_file)

        return symlink_path

    # Handle single directory/file symlink (standard case)
    if symlink_path.is_symlink():
        symlink_path.unlink()
    assert (
        not symlink_path.exists()
    ), f"symlink location: {symlink_path} has a non-symlink there."
    symlink_path.symlink_to(weights_dir)
    return symlink_path


def setup_mesh_device(mesh_device_config):
    """Setup mesh device with proper device parameters using conftest.py approach"""
    try:
        # Parse mesh device config
        mesh_config = eval(mesh_device_config)  # e.g., (8, 4)
        print(f"Initializing mesh device with config: {mesh_config}")
        
        # Configure device parameters (hardcoded like text_demo.py)
        device_params = {
            "trace_region_size": 140280832,
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1344544,
            "fabric_config": True,
        }
        
        print(f"Device parameters: {device_params}")
        
        # Process device parameters like get_updated_device_params() in tests/scripts/common.py
        updated_device_params = device_params.copy()
        
        dispatch_core_axis = updated_device_params.pop("dispatch_core_axis", None)
        dispatch_core_type = updated_device_params.pop("dispatch_core_type", None)
        
        if ttnn.device.is_blackhole() and dispatch_core_axis == ttnn.DispatchCoreAxis.ROW:
            logger.warning("blackhole arch does not support DispatchCoreAxis.ROW, using DispatchCoreAxis.COL instead.")
            dispatch_core_axis = ttnn.DispatchCoreAxis.COL
        
        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
        updated_device_params["dispatch_core_config"] = dispatch_core_config
        
        # Extract fabric_config and reliability_mode like conftest.py
        fabric_config = updated_device_params.pop("fabric_config", None)
        reliability_mode = updated_device_params.pop("reliability_mode", None)
        
        # Follow the same pattern as the mesh_device fixture in conftest.py
        device_ids = ttnn.get_device_ids()
        
        if isinstance(mesh_config, tuple):
            grid_dims = mesh_config
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if not ttnn.using_distributed_env() and num_devices_requested > len(device_ids):
                print(f"Requested {num_devices_requested} devices but only {len(device_ids)} available")
                return None, None
            mesh_shape = ttnn.MeshShape(*grid_dims)
        else:
            num_devices_requested = min(mesh_config, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)
        
        print(f"Opening mesh device with shape: {mesh_shape}")
        
        # Set fabric config before opening mesh device (like conftest.py set_fabric function)
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        
        # Use the same approach as conftest.py mesh_device fixture
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
        
        print(f"Mesh device initialized successfully with {mesh_device.get_num_devices()} devices")
        return mesh_device, fabric_config
        
    except Exception as e:
        print(f"Failed to initialize mesh device: {e}")
        print("Make sure you're running in a proper tt-metal environment with TT devices available.")
        return None, None


def reset_kv_cache(model):
    """Reset KV cache between batches - EXACT same as text_demo.py"""
    model.switch_mode("prefill")
    for layer in model.layers:
        k_cache, v_cache = layer.attention.layer_past
        k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
        v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)


def pad_batch_tensors(current_pos, page_table, out_tok, batch_size):
    """Pad tensors for batch size - EXACT same logic as text_demo.py"""
    if batch_size == 1:
        # pad current_pos to 32 with -1s
        current_pos = torch.nn.functional.pad(current_pos, (0, 32 - current_pos.shape[0]), value=-1)
        # pad page_table to 32 with 0s
        if page_table is not None:
            page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 32 - page_table.shape[0]), value=0)
    
    if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
        out_tok = out_tok.repeat(32, 1)
    
    return current_pos, page_table, out_tok


def run_inference_batch(
    batch_idx,
    input_prompts,
    model,
    model_args,
    generator,
    tokenizer,
    page_table,
    tt_kv_cache,
    page_params,
    sampling_params,
    batch_size,
    max_generated_tokens,
    max_seq_len,
    instruct,
    enable_trace,
    stop_at_eos,
    use_paged_kv_cache
):
    """
    Run inference for a single batch - contains the EXACT same logic as text_demo.py
    from "logger.info(f'Processing batch {batch_idx}')" onwards
    """
    logger.info(f"Processing batch {batch_idx}")
    batch_start_time = time.time()
    
    # Preprocess initial prompt inputs
    try:
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            tokenizer,
            [model_args],
            instruct,
            max_generated_tokens,
        )
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None, 0
    
    max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
    
    assert (
        max_generated_tokens + max_encoded_prompt_len <= max_seq_len
    ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"
    
    batch_size_per_device_group = (
        32 if batch_size == 32 else 1
    )  # This is a workaround until page table needs to know that attention is DP
    
    if use_paged_kv_cache:
        paged_cache_max_seq_len = (
            page_params["page_block_size"] * page_params["page_max_num_blocks"] / batch_size_per_device_group
        )
        assert (
            max_generated_tokens + max_encoded_prompt_len <= paged_cache_max_seq_len
        ), f"max_generated_tokens ({max_generated_tokens}) needs to be <= than paged_cache_max_seq_len ({paged_cache_max_seq_len})"
    
    # when doing repeating batches, set kv-caches to zero, to avoid context leaking
    if batch_idx != 0:
        reset_kv_cache(model)
    
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)
    
    if batch_idx == 0:
        logger.info("Starting prefill warmup...")
        try:
            toks = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                enable_trace=enable_trace,
            )
        except Exception as e:
            logger.error(f"Error during prefill warmup: {str(e)}")
            raise e
        logger.info("Finished prefill warmup")
    
    logger.info(f"Starting prefill...")
    
    try:
        toks = generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            enable_trace=enable_trace,
        )
    except Exception as e:
        logger.error(f"Error during prefill: {str(e)}")
        raise e
    
    # Save prefill token
    prefilled_token = toks.view(-1, 1)
    logger.info(f"Prefill finished")
    
    # Keep track of generated outputs to print out every iteration
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(prefilled_token[user].item())
        all_outputs[user].append(user_tok)
    
    # Keeps track when a user reaches EoD token
    user_done = [False] * batch_size
    
    device_sampling_params = SamplingParams(
        temperature=sampling_params["temperature"], top_k=32, top_p=sampling_params["top_p"]
    )
    
    # Initial positions
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
    out_tok = prefilled_token
    
    # Pad tensors based on batch size - EXACT same as text_demo.py
    current_pos, page_table, out_tok = pad_batch_tensors(current_pos, page_table, out_tok, batch_size)
    
    # Start decoding
    iteration = 0
    users_decoding = True
    
    try:
        model.switch_mode("decode")
    except Exception as e:
        logger.error(f"Error switching to decode mode: {str(e)}")
        model.tt_ccl.close()
    
    logger.info(f"Starting decode loop from positions: {decoding_pos}")
    
    # Track decode outputs for proper synchronization
    read_events = []
    tt_out_toks = []
    
    while users_decoding:
        # Run decode forward
        try:
            tt_out_tok, read_event = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                sampling_params=device_sampling_params,
                reset_inputs=iteration == 0,
            )
            read_events.append(read_event)
            tt_out_toks.append(tt_out_tok)
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}")
            break
        
        # Process output from previous iteration (skip first iteration)
        if iteration > 0:
            ttnn.event_synchronize(read_events.pop(0)[0])
            tt_out_tok = generator.process_decode_output_host(tt_out_toks.pop(0))
            
            out_tok = tt_out_tok
            if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
                out_tok = out_tok.repeat(32, 1)
            
            # Save output token to print out later
            for user in range(batch_size):
                user_tok = out_tok.tolist()[user]
                if (
                    user_tok not in tokenizer.stop_tokens and user_done[user] == False
                ):  # Read until an eos token
                    all_outputs[user].append(user_tok)
                else:
                    if stop_at_eos:
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False
        
        current_pos += 1
        iteration += 1
        
        # Upper limit of generated tokens for each user
        if users_decoding:
            users_decoding = iteration < max_generated_tokens
    
    # Process final output
    if read_events:
        ttnn.event_synchronize(read_events.pop(0)[0])
        tt_out_tok = generator.process_decode_output_host(tt_out_toks.pop(0))
        
        # Save final output token
        for user in range(batch_size):
            user_tok = tt_out_tok.tolist()[user]
            if user_tok not in tokenizer.stop_tokens:
                all_outputs[user].append(user_tok)
    
    batch_time = time.time() - batch_start_time
    
    # Return results
    batch_results = []
    for i, output in enumerate(all_outputs):
        text = tokenizer.decode(output)
        prompt_including_assistant_tags = tokenizer.decode(
            model_args.encode_prompt(input_prompts[i], instruct=instruct)
        )
        text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
        
        result = {
            "batch_idx": batch_idx,
            "user_idx": i,
            "tokens_generated": iteration,
            "processing_time": batch_time / batch_size,  # Time per user
            "output": text_after_prompt.strip()
        }
        batch_results.append(result)
    
    logger.info(f"Batch {batch_idx} completed in {batch_time:.2f} seconds")
    logger.info(f"Generated {iteration} tokens per user")
    
    return batch_results, iteration


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Batch Local LLM Inference Script v3 - Uses EXACT same inference approach as text_demo.py",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of users in a batch (default: 1, supports 1/2/4/8/16/32)'
    )
    
    parser.add_argument(
        '--repeat_batches',
        type=int,
        default=10,
        help='Number of times to repeat the batch processing (default: 3)'
    )
    
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=128 * 1024,
        help='Maximum context length supported by the model (default: 128 * 1024)'
    )
    
    parser.add_argument(
        '--max_generated_tokens',
        type=int,
        default=1024,
        help='Maximum number of tokens to generate for each request (default: 128)'
    )
    
    parser.add_argument(
        '--mesh_device_config',
        type=str,
        default='(8,4)',
        help='Mesh device configuration as tuple string (default: "(8,4)")'
    )
    
    parser.add_argument(
        '--prompt_file',
        type=str,
        default='sample_prompts/repro_prompt_ISL_3175.json',
        help='Path to JSON file containing the prompt (default: "sample_prompts/repro_prompt_ISL_3175.json")'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Starting batch inference with {args.batch_size} users per batch × {args.repeat_batches} repeat batches")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Max generated tokens: {args.max_generated_tokens}")
    print(f"Results will be saved to: {JSON_FILENAME}")
    print("=" * 60)
    
    # Initialize mesh device
    mesh_device, fabric_config = setup_mesh_device(args.mesh_device_config)
    if mesh_device is None:
        return 1
    
    # Create model components (following text_demo.py exactly)
    instruct = True
    optimizations = LlamaOptimizations.performance
    num_layers = 80
    dummy_weights = not instruct
    page_params = {"page_block_size": 64, "page_max_num_blocks": 2048}
    dtype = ttnn.bfloat8_b
    use_paged_kv_cache = True
    enable_trace = True
    stop_at_eos = False  # Don't stop at EOS to generate full response
    
    # Load prompt from JSON file
    logger.info(f"Loading prompt from: {args.prompt_file}")
    raw_prompt = load_prompt_from_file(args.prompt_file)
    
    if not raw_prompt:
        logger.error(f"Failed to load prompt from {args.prompt_file}")
        return 1
    
    # Parse the raw prompt to extract system and user messages
    # The prompt appears to be in chat format with special tokens
    system_message = None
    user_message = None
    
    # Look for system message between <|start_header_id|>system<|end_header_id|> and <|eot_id|>
    system_match = re.search(r'<\|start_header_id\|>system<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', raw_prompt, re.DOTALL)
    if system_match:
        system_message = system_match.group(1).strip()
    
    # Look for user message between <|start_header_id|>user<|end_header_id|> and <|eot_id|>
    user_match = re.search(r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', raw_prompt, re.DOTALL)
    if user_match:
        user_message = user_match.group(1).strip()
    
    # If we couldn't parse the chat format, use the raw prompt as user message
    if not system_message and not user_message:
        logger.warning("Could not parse chat format, using raw prompt as user message")
        user_message = raw_prompt
    
    # Update REQUEST_PAYLOAD with parsed messages
    REQUEST_PAYLOAD["messages"] = []
    if system_message:
        REQUEST_PAYLOAD["messages"].append({
            "role": "system",
            "content": system_message
        })
    if user_message:
        REQUEST_PAYLOAD["messages"].append({
            "role": "user", 
            "content": user_message
        })
    
    logger.info(f"Loaded prompt with {len(REQUEST_PAYLOAD['messages'])} messages")
    
    # Create prompts for batch (repeat the same prompt for each user)
    if system_message and user_message:
        prompt = f"{system_message}\n\n{user_message}"
    else:
        prompt = user_message
    
    # Create list of prompts for batch_size users
    input_prompts = [prompt] * args.batch_size
    
    # To simulate repeat_batches, we'll create multiple sets of prompts
    repeat_batch_prompts = []
    for i in range(args.repeat_batches):
        repeat_batch_prompts.append(input_prompts)
    
    model_args, model, page_table, tt_kv_cache = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=args.batch_size,
        optimizations=optimizations,
        max_seq_len=args.max_seq_len,
        num_layers=num_layers,
        dummy_weights=dummy_weights,
        page_params=page_params,
        dtype=dtype,
        use_paged_kv_cache=use_paged_kv_cache,
        prefill_profile=False,
    )
    
    model_args.tokenizer = Tokenizer(model_args.tokenizer_path)
    tokenizer = model_args.tokenizer
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)
    
    # Sampling parameters
    sampling_params = {"temperature": REQUEST_PAYLOAD.get("temperature", 0.0), "top_p": 0.08}
    
    # Prepare initial metadata
    initial_metadata = {
        "timestamp": TIMESTAMP,
        "batch_size": args.batch_size,
        "repeat_batches": args.repeat_batches,
        "max_generated_tokens": args.max_generated_tokens,
        "model": REQUEST_PAYLOAD["model"],
        "start_time": time.time()
    }
    
    logger.info("Starting inference...")
    overall_start_time = time.time()
    
    # Run inference for each batch using the EXACT same logic as text_demo.py
    for batch_idx, input_prompts in enumerate(repeat_batch_prompts):
        batch_results, tokens_generated = run_inference_batch(
            batch_idx=batch_idx,
            input_prompts=input_prompts,
            model=model,
            model_args=model_args,
            generator=generator,
            tokenizer=tokenizer,
            page_table=page_table,
            tt_kv_cache=tt_kv_cache,
            page_params=page_params,
            sampling_params=sampling_params,
            batch_size=args.batch_size,
            max_generated_tokens=args.max_generated_tokens,
            max_seq_len=args.max_seq_len,
            instruct=instruct,
            enable_trace=enable_trace,
            stop_at_eos=stop_at_eos,
            use_paged_kv_cache=use_paged_kv_cache
        )
        
        if batch_results:
            # Write results immediately after each batch completion
            metadata_for_batch = initial_metadata if batch_idx == 0 else None
            is_final_batch = (batch_idx == len(repeat_batch_prompts) - 1)
            
            # Prepare final metadata if this is the last batch
            final_metadata = None
            if is_final_batch:
                total_time = time.time() - overall_start_time
                final_metadata = {
                    "total_time": total_time,
                    "avg_time_per_batch": total_time / args.repeat_batches if args.repeat_batches > 0 else 0,
                    "end_time": time.time(),
                    "batches_completed": len(repeat_batch_prompts)
                }
            
            write_batch_results_incremental(
                batch_results, 
                metadata=metadata_for_batch,
                final_metadata=final_metadata,
                is_final=is_final_batch
            )
            logger.info(f"Wrote batch {batch_idx} results to {JSON_FILENAME}")
    
    total_time = time.time() - overall_start_time
    
    print("=" * 60)
    print(f"All batches completed in {total_time:.2f} seconds")
    print(f"Results saved to: {JSON_FILENAME}")
    
    # Cleanup - Following the exact pattern from conftest.py
    try:
        # First, close the model's CCL if it exists
        if 'model' in locals() and hasattr(model, 'tt_ccl'):
            model.tt_ccl.close()
            print("Model CCL closed successfully")
        
        # Delete references to generator and model to ensure traces are cleaned up
        if 'generator' in locals():
            del generator
        if 'model' in locals():
            del model
        if 'model_args' in locals():
            del model_args
        if 'tt_kv_cache' in locals():
            del tt_kv_cache
        if 'page_table' in locals():
            del page_table
        if 'tokenizer' in locals():
            del tokenizer
        
        # Force garbage collection to clean up any remaining references
        import gc
        gc.collect()
        
        # Then close mesh device following conftest.py pattern
        if 'mesh_device' in locals() and mesh_device is not None:
            # Close submeshes first, then the main mesh device
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
            print("Mesh device closed successfully")
            
            # Reset fabric config after closing mesh device (like conftest.py)
            if 'fabric_config' in locals() and fabric_config:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                print("Fabric config reset to DISABLED")
            
            # Delete the mesh_device reference
            del mesh_device
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return 0


def set_up_environment():
    """Set up environment variables (same as original script)"""
    hf_model = os.getenv("HF_MODEL")
    llama_dir = os.getenv("LLAMA_DIR")
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_PATH"))
    print(f"original HF_MODEL:={hf_model}")
    print(f"original LLAMA_DIR:={llama_dir}")
    print(f"original MODEL_WEIGHTS_PATH:={weights_dir}")
    if weights_dir:
        symlinks_dir = script_path = Path(__file__).parent / "model_file_symlinks_map"
        print(f"creating symlinks_dir:={symlinks_dir}")
        symlinks_dir.mkdir(parents=True, exist_ok=True)
        if "Llama" in str(weights_dir):
            model_dir_name = "Llama-3.3-70B-Instruct"
            # the mapping in: models/tt_transformers/tt/model_spec.py
            # uses e.g. Llama3.2 instead of Llama-3.2
            model_dir_name = model_dir_name.replace("Llama-", "Llama")
            file_symlinks_map = {}
            
            llama_dir = create_model_symlink(
                symlinks_dir,
                model_dir_name,
                weights_dir,
                file_symlinks_map=file_symlinks_map,
            )
            print(f"new LLAMA_DIR:={llama_dir}")
            print(f"new HF_MODEL:={None}")
            os.environ["LLAMA_DIR"] = str(llama_dir)
            if "HF_MODEL" in os.environ:
                del os.environ["HF_MODEL"]
        else:
            print(f"no symlinks needed for {weights_dir}")
            os.environ["HF_MODEL"] = str(weights_dir)
            print(f"new HF_MODEL:={weights_dir}")


if __name__ == "__main__":
    set_up_environment()
    import sys
    sys.exit(main())