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
"""

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from datetime import datetime
import os
import argparse
from pathlib import Path
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


# Sample request payload (same as v2)
REQUEST_PAYLOAD = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "temperature": 0.0,
    "max_tokens": 1000,
    "stream": False,
    "messages": [
        {
            "role": "system",
            "content": """
                        # COMPREHENSIVE PHYLOGENETIC AND MORPHOMETRIC ANALYSIS MISSION
                        Execute a complete multi-dimensional taxonomic assessment encompassing ALL botanical structures, morphological characteristics, phylogenetic indicators, anatomical features, developmental patterns, ecological adaptations, biochemical markers, reproductive strategies, and evolutionary relationships present in the provided specimen documentation with absolute precision, systematic rigor, phylogenetic accuracy, and comprehensive taxonomic consistency across all hierarchical classification levels.

                        # ADVANCED BOTANICAL CLASSIFICATION FRAMEWORK

                        ## Primary Botanical Structures (Macroscopic Level)
                        Primary botanical structures represent major organ systems and anatomical components visible through standard morphological examination techniques.
                        - Utilized for systematic phylogenetic reconstruction, cladistic analysis, and comparative morphological studies
                        - Encompasses vegetative, reproductive, and specialized anatomical architectures
                        - Includes both determinate and indeterminate growth patterns with associated developmental trajectories
                        - Examples: "cauline architecture", "radicle systems", "cotyledonary arrangements", "meristematic zones", "vascular cambium", "periderm layers", "secretory ducts"

                        ## Secondary Morphological Characteristics (Microscopic Level)
                        Secondary morphological characteristics define ultrastructural features and cellular-level organizational patterns that determine taxonomic placement and evolutionary relationships.
                        - Provides precise morphometric data for phylogenetic reconstruction and systematic classification
                        - Establishes "definitive, quantifiable morphological parameters for taxonomic delimitation"
                        - Represents specialized terminology for anatomical precision and systematic identification
                        - Examples: "bulliform cells", "sclerenchymatous tissue", "aerenchyma formation", "crystalliferous idioblasts", "mucilaginous secretions", "reticulate venation patterns"

                        ## Tertiary Phylogenetic Indicators (Molecular-Morphological Interface)
                        Tertiary phylogenetic indicators encompass morphological expressions of underlying genetic and biochemical processes that reflect evolutionary history and systematic relationships.
                        - Demonstrates evolutionary constraints and adaptive radiations within taxonomic lineages
                        - Reveals synapomorphic character states and plesiomorphic retention patterns
                        - Indicates convergent evolution versus homologous character development
                        - Examples: "anthocyanin distribution patterns", "terpene-producing glandular structures", "C4 photosynthetic anatomy", "crassulacean acid metabolism adaptations"

                        ## Quaternary Ecological Adaptations (Environmental Interface)
                        Quaternary ecological adaptations represent morphological modifications and physiological specializations that reflect environmental pressures and habitat-specific evolutionary responses.
                        - Demonstrates phenotypic plasticity and genotype-environment interactions
                        - Reveals adaptive syndrome complexes and ecological niche specialization
                        - Indicates coevolutionary relationships and community-level interactions
                        - Examples: "xeromorphic adaptations", "hydrophytic modifications", "mycorrhizal association structures", "allelopathic compound production sites"

                        # COMPLEX INTERRELATIONSHIPS AND HIERARCHICAL DEPENDENCIES
                        Primary structures provide the foundational architecture; secondary characteristics define the detailed morphometric parameters; tertiary indicators reveal phylogenetic positioning; quaternary adaptations demonstrate ecological specialization. Each level requires specific analytical approaches, and comprehensive analysis must integrate all hierarchical levels simultaneously while maintaining taxonomic precision and evolutionary context.

                        # COMPREHENSIVE EXCLUSION CRITERIA FOR TAXONOMIC ANALYSIS

                        1. **Vernacular Horticultural Terminology**: Non-technical terms commonly employed in amateur botanical contexts without specialized morphological precision
                        - Diagnostic Test: "Would this term appear in basic gardening literature without technical definition?"
                        - Examples: "pretty flowers", "green leaves", "tall plants", "thick stems", "small fruits", "nice smell", "fast growth", "easy care", "popular variety", "common garden plant"

                        2. **Taxonomic Nomenclature and Geographic Designations**: Formal scientific names, cultivar designations, institutional affiliations, geographic localities, and personal attributions
                        - Diagnostic Test: "Does this represent a formal taxonomic entity, location, or attribution rather than morphological description?"
                        - Examples: "International Botanical Research Institute", "Himalayan Collection Center", "Professor Anderson's Herbarium", "Botanical Survey Expedition", "Alpine Research Facility", "Systematic Biology Laboratory"

                        3. **Scientific Literature and Reference Citations**: Publication titles, journal names, monographic works, taxonomic treatments, and bibliographic references
                        - Diagnostic Test: "Is this a formal publication title or bibliographic reference rather than morphological terminology?"
                        - Examples: "Systematic Botany Quarterly", "Phylogenetic Analysis Methods", "Comparative Morphology Handbook", "Taxonomic Revision Series", "Botanical Monograph Collection", "Systematic Biology References"

                        4. **Quantitative Measurements and Numerical Data**: Pure numerical values, statistical parameters, measurement units, and mathematical expressions without morphological context
                        - Diagnostic Test: "Does this represent raw data rather than morphological terminology?"
                        - Examples: "15.7 centimeters", "pH 6.8", "temperature 23°C", "humidity 65%", "elevation 1200m", "statistical significance p<0.05"

                        # ADVANCED ACCEPTANCE CRITERIA FOR MORPHOLOGICAL ANALYSIS

                        ## Primary Structure Identification Protocol
                        **Diagnostic Test**: "Does this term represent a discrete anatomical entity with specific physiological function and taxonomic significance?"
                        - Organ-level anatomical components with defined developmental origins
                        - Tissue-level organizational units with specialized cellular architecture
                        - Cellular-level structures with distinctive morphological characteristics
                        - Subcellular components with taxonomically relevant features
                        - Biochemical structures with morphological expression and systematic importance

                        ## Secondary Characteristic Classification Protocol
                        **Diagnostic Test**: "Does this term provide precise morphometric description with taxonomic diagnostic value?"
                        - Specialized morphological descriptors with quantitative precision
                        - Professional systematic terminology with established usage in taxonomic literature
                        - Technical morphological attributes with comparative analytical value
                        - Phylogenetic character states with evolutionary significance
                        - Ecological adaptation descriptors with morphological basis and systematic relevance

                        ## Tertiary Integration Assessment Protocol
                        **Diagnostic Test**: "Does this term integrate multiple analytical levels with comprehensive taxonomic implications?"
                        - Multi-dimensional morphological concepts spanning structural and functional domains
                        - Systematic terminology bridging morphological and phylogenetic analytical frameworks
                        - Comparative descriptors enabling cross-taxonomic morphological analysis
                        - Evolutionary morphological concepts with developmental and systematic significance

                        # COMPREHENSIVE ANALYTICAL METHODOLOGY

                        ## Phase I: Preliminary Morphological Survey
                        1. Conduct exhaustive lexical analysis from document initiation through completion, examining every morphological term and technical descriptor
                        2. Evaluate complex morphological phrases and compound technical terminology as integrated analytical units
                        3. Apply hierarchical diagnostic protocols to each identified term and phrase combination
                        4. Establish preliminary morphological categories based on structural complexity and taxonomic significance
                        5. Document questionable cases requiring additional analytical consideration
                        6. Eliminate redundant identifications while preserving morphological precision

                        ## Phase II: Advanced Taxonomic Classification
                        1. Cross-reference identified terms with established systematic terminology databases
                        2. Verify morphological accuracy through comparative anatomical analysis
                        3. Assess phylogenetic significance and evolutionary implications of identified characteristics
                        4. Evaluate ecological context and adaptive significance of morphological features
                        5. Integrate multi-level analytical results into comprehensive taxonomic assessment
                        6. Validate final classifications through systematic review protocols

                        ## Phase III: Comprehensive Integration and Verification
                        1. Synthesize all analytical levels into unified morphological assessment
                        2. Verify taxonomic consistency across all hierarchical classification levels
                        3. Confirm morphological precision and systematic accuracy of all identifications
                        4. Validate evolutionary and ecological interpretations of morphological data
                        5. Ensure comprehensive coverage of all morphological elements present in source material
                        6. Finalize integrated taxonomic analysis with complete morphological documentation

                        # ADVANCED COMPOUND TERMINOLOGY ANALYSIS

                        - Complex morphological phrases require integrated analytical treatment as complete systematic units
                        - Example: analyze "xeromorphic sclerophyllous leaf architecture with crassulacean photosynthetic adaptations" as complete morphological syndrome rather than individual components
                        - Multiple related terms such as "stomatal complexes", "stomatal distribution patterns", and "stomatal developmental sequences" should all receive independent analytical treatment
                        - When primary morphological structures are identified, conduct comprehensive analysis of associated modifying terminology and contextual descriptors
                        - Integrate morphological terminology with ecological, developmental, and phylogenetic contextual information

                        # CRITICAL ANALYTICAL REQUIREMENTS AND QUALITY STANDARDS

                        - Restrict analysis exclusively to morphological terminology present in provided documentation; ABSOLUTELY PROHIBIT fabrication or supplementation of analytical content
                        - MANDATORY requirement for complete morphological coverage; systematic failure to identify any relevant terminology is unacceptable
                        - Utilize EXCLUSIVELY "structure", "characteristic", "phylogenetic_indicator", or "ecological_adaptation" classifications in type designation fields
                        - Eliminate duplicate morphological identifications in final output while preserving case-sensitive morphological precision
                        - Output formatting must conform strictly to JSON specifications without markdown formatting, explanatory text, or supplementary documentation
                        - Maintain absolute distinction between anatomical entities (structures), descriptive attributes (characteristics), evolutionary markers (phylogenetic indicators), and environmental specializations (ecological adaptations)
                        - STRICTLY PROHIBIT classification of formal taxonomic names, geographic designations, or bibliographic references as morphological terminology

                        # ADVANCED MORPHOLOGICAL ASSESSMENT PROTOCOLS

                        ## Structural Complexity Analysis
                        - Evaluate morphological complexity across multiple organizational levels (molecular, cellular, tissue, organ, organism)
                        - Assess developmental trajectories and ontogenetic morphological changes
                        - Analyze functional morphology and structure-function relationships
                        - Investigate comparative morphological patterns and evolutionary trends

                        ## Phylogenetic Character Assessment
                        - Identify synapomorphic character states with taxonomic diagnostic value
                        - Evaluate morphological homology versus convergent similarity
                        - Assess character state polarization and evolutionary directionality
                        - Analyze morphological constraint patterns and developmental limitations

                        ## Ecological Morphology Integration
                        - Evaluate adaptive morphological syndromes and functional complexes
                        - Assess environmental correlation patterns and habitat specialization indicators
                        - Analyze coevolutionary morphological modifications and community interaction effects
                        - Investigate phenotypic plasticity patterns and environmental response mechanisms

                        # COMPREHENSIVE OUTPUT SPECIFICATION

                        
                        Final analytical output must conform to strict JSON formatting requirements without any supplementary documentation, markdown formatting, or explanatory content. ABSOLUTELY PROHIBIT usage of ``` delimiters, "json" designations, or any formatting markers preceding or following JSON content!
                        Deliver results exclusively in pure JSON format without markdown formatting such as "```" or "json" designations - provide only clean JSON output.

                        {
                            "comprehensive_morphological_analysis": [
                                {
                                    "morphological_term": "identified_morphological_terminology",
                                    "analytical_category": "structure/characteristic/phylogenetic_indicator/ecological_adaptation",
                                    "taxonomic_significance": "primary/secondary/tertiary/quaternary",
                                    "morphological_complexity": "simple/compound/integrated/systematic"
                                }
                            ]
                        }

                        If comprehensive analysis yields no morphological terminology:
                        {
                            "comprehensive_morphological_analysis": []
                        }
                        """
        },
        {
            "role": "user",
            "content": "COMPREHENSIVE PHYLOGENETIC SPECIMEN DOCUMENTATION FOR MULTI-DIMENSIONAL MORPHOLOGICAL ANALYSIS: Section 7. INTEGRATED MORPHOLOGICAL ARCHITECTURE AND EVOLUTIONARY ADAPTIVE SYNDROME COMPLEXES - The taxonomic specimen demonstrates extraordinary xeromorphic sclerophyllous foliar architecture characterized by distinctive oblanceolate-spatulate primary photosynthetic organs exhibiting pronounced crenate-serrate marginal dentition with specialized hydathode terminations, measuring 18.7-23.4 cm in maximum longitudinal dimension and 4.2-6.8 cm in transverse width parameters (eighteen point seven to twenty-three point four centimeters longitudinally, four point two to six point eight centimeters transversely). The specimen's complex bulliform cell arrangements within adaxial epidermal tissues, combined with specialized aerenchymatous parenchyma distribution patterns and crystalliferous idioblast positioning, create sophisticated water-storage mechanisms and osmotic regulation systems. Within this comprehensive phylogenetic classification framework, the specimen's integrated cauline architecture, decussate phyllotactic arrangements, and multi-layered environmental adaptation syndrome complexes demonstrate remarkable phenotypic plasticity accommodating extreme seasonal phenological fluctuations, circadian photosynthetic optimization cycles, crassulacean acid metabolism transitions, and complex allelopathic biochemical production processes, thereby eliminating requirements for additional specialized morphological adaptation mechanisms for these intricate physiological and ecological functions. The specimen exhibits compound cymose inflorescence structures with specialized nectariferous disc arrangements and complex pollination syndrome adaptations as documented by the International Systematic Biology Research Consortium including: (a) — 6 (six) monthly anthesis cycle achievements correlated with short-term entomophilous pollination efficiency optimization systems featuring maximum 6 (six) monthly specialized nectar production volumes with enhanced amino acid concentrations, (b) — comprehensive annual reproductive target fulfillment connected to integrated short-term and long-term pollination syndrome optimization systems with maximum 12 (twelve) monthly equivalent nectar production capacities incorporating specialized volatile organic compound emissions, (c) — extended long-term evolutionary adaptation trajectory plans encompassing 3 (three) complete annual phenological cycles with maximum 3 (three) yearly integrated metabolic efficiency enhancement rates achieving 40% (forty percent) optimization potential through specialized biochemical pathway modifications and morphological plasticity expressions. Additionally, the specimen demonstrates complex mycorrhizal association structures with specialized root hair modifications, enhanced cortical aerenchyma development, and sophisticated nutrient uptake mechanisms involving specialized transfer cells and symplastic transport pathways. The specimen's reproductive organs exhibit remarkable heterostyly with specialized anther positioning, complex stigmatic surface modifications, and intricate pollen presentation mechanisms designed for optimal cross-pollination success. To eliminate taxonomic ambiguity and ensure systematic precision, when the specimen achieves partial or complete reproductive success through short-term entomophilous pollination efficiency optimization systems as determined by the International Systematic Biology Research Consortium protocols for 6 (six) monthly and comprehensive annual reproductive target parameters, the total morphological and biochemical enhancement potential available to the specimen through phenotypic plasticity expressions and adaptive syndrome modifications will not exceed the specimen's established 12 (twelve) monthly baseline integrated metabolic efficiency rates, as measured through standardized photosynthetic capacity assessments, specialized enzyme activity quantification, and comprehensive secondary metabolite production analyses."
        }
    ]
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
    
    # Extract messages from our prompt
    messages = REQUEST_PAYLOAD["messages"]
    system_message = None
    user_message = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        elif msg["role"] == "user":
            user_message = msg["content"]
    
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