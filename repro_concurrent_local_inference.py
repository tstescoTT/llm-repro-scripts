#!/usr/bin/env python3
"""
Concurrent Local LLM Inference Script
Makes configurable concurrent requests to a locally loaded LLM model and saves results to a timestamped JSON file.
Supports multiple loops of concurrent requests with configurable timeout.
Based on repro_concurrent_llm_requests.py but uses local model inference instead of HTTP requests.

CLI Options:
  --concurrency: Maximum number of concurrent requests (default: 32)
  --loops: Number of times to loop over (concurrency) requests (default: 3)
  --timeout: Timeout for each request in seconds (default: no timeout)
  --max_seq_len: Maximum context length supported by the model (default: 128 * 1024)
  --batch_size: Number of users in a batch (default: 1)
  --max_generated_tokens: Maximum number of tokens to generate (default: 128)
"""

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from datetime import datetime
import uuid
import sys
import os
import argparse
import signal
from pathlib import Path
import pytest
import torch
import ttnn
from loguru import logger

from models.demos.llama3_70b_galaxy.tt.generator import Generator, SamplingParams
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.tt_transformers.tt.common import (
    preprocess_inputs_prefill,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import TtModelArgs

# Generate timestamped filename at startup
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = "output"
JSON_FILENAME = os.path.join(OUTPUT_DIR, f"concurrent_local_inference_{TIMESTAMP}.json")

# Global counter for tracking written results
results_written_count = 0
results_written_lock = None

# Global model components (will be initialized once)
model_components = {
    'model': None,
    'generator': None,
    'tokenizer': None,
    'model_args': None,
    'page_table': None,
    'tt_kv_cache': None,
    'mesh_device': None
}

# Sample request payload (similar to the original script)
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
    """Helper function to create and manage model symlinks.

    Args:
        symlinks_dir: Directory to store symlinks
        model_name: Model name to use for the symlink
        weights_dir: Path to the model weights
        file_symlinks_map: Dict of {target_file: source_file} for creating file-specific symlinks

    Returns:
        Path to the created symlink or directory
    """
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


class RequestResult:
    def __init__(self, request_id, success=False, response=None, error=None, processing_time=0):
        self.request_id = request_id
        self.timestamp = datetime.now().isoformat()
        self.success = success
        self.response = response
        self.error = error
        self.processing_time = processing_time

def write_result_to_incremental_json(result, json_lock):
    """Write a single result to the incremental JSON file immediately"""
    global results_written_count
    
    try:
        request_data = {
            "request_id": result.request_id,
            "timestamp": result.timestamp,
            "success": result.success,
            "processing_time": result.processing_time
        }
        
        if result.success:
            request_data["response"] = result.response
            request_data["error"] = None
        else:
            request_data["response"] = None
            request_data["error"] = result.error
        
        # Write to JSON file with thread safety
        with json_lock:
            with results_written_lock:
                # Check if this is the first result
                if results_written_count == 0:
                    # Write opening bracket and first object
                    with open(JSON_FILENAME, 'w', encoding='utf-8') as f:
                        f.write('[\n')
                        f.write(json.dumps(request_data, ensure_ascii=False, indent=2))
                        f.flush()
                else:
                    # Append comma and next object
                    with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                        f.write(',\n')
                        f.write(json.dumps(request_data, ensure_ascii=False, indent=2))
                        f.flush()
                
                results_written_count += 1
                
    except Exception as e:
        print(f"Error writing result to incremental JSON: {e}")

def close_json_array():
    """Close the JSON array"""
    try:
        if results_written_count > 0:
            with open(JSON_FILENAME, 'a', encoding='utf-8') as f:
                f.write('\n]\n')
                f.flush()
            print(f"Closed JSON file: {JSON_FILENAME}")
    except Exception as e:
        print(f"Error closing JSON array: {e}")

def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    num_layers,
    dummy_weights,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=False,
    prefill_profile=False,
):
    """Create and initialize the TT model components (same as text_demo.py)"""
    tt_model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=32,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )
    # When running running prefill-only profile, run just 1 layer
    tt_model_args.n_layers = num_layers if not prefill_profile else 1

    state_dict = tt_model_args.load_state_dict()
    page_table = None
    paged_attention_config = None
    tt_kv_cache = None

    if use_paged_kv_cache:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            max_batch_size, paged_attention_config.max_num_blocks // max_batch_size
        )

    model = TtTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        mode="prefill",
        enable_prefetcher_performance_mode=True,
    )

    if use_paged_kv_cache:
        tt_kv_cache = [l.attention.layer_past for l in model.layers]

    return tt_model_args, model, page_table, [tt_kv_cache]

def initialize_model_components(mesh_device, max_seq_len, max_generated_tokens, batch_size=1):
    """Initialize the model components once for all threads to share"""
    global model_components
    
    try:
        instruct = True
        optimizations = LlamaOptimizations.performance
        num_layers = 80
        dummy_weights = not instruct
        page_params = {"page_block_size": 64, "page_max_num_blocks": 2048}
        dtype = ttnn.bfloat8_b
        use_paged_kv_cache = True
        
        model_args, model, page_table, tt_kv_cache = create_tt_model(
            mesh_device,
            instruct=instruct,
            max_batch_size=batch_size,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
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
        
        model_components['model_args'] = model_args
        model_components['model'] = model
        model_components['page_table'] = page_table
        model_components['tt_kv_cache'] = tt_kv_cache
        model_components['tokenizer'] = tokenizer
        model_components['generator'] = generator
        model_components['mesh_device'] = mesh_device
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model components: {e}")
        return False

def make_local_inference_request(request_id, results_list, lock, json_lock, timeout=None, max_generated_tokens=128):
    """Make a single local inference request and store the result"""
    start_time = time.time()
    
    try:
        # Extract the user message from the request payload
        messages = REQUEST_PAYLOAD["messages"]
        user_message = None
        system_message = None
        
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "system":
                system_message = msg["content"]
        
        if not user_message:
            raise ValueError("No user message found in request payload")
        
        # Get model components
        tokenizer = model_components['tokenizer']
        generator = model_components['generator']
        model = model_components['model']
        page_table = model_components['page_table']
        tt_kv_cache = model_components['tt_kv_cache']
        
        if not all([tokenizer, generator, model]):
            raise ValueError("Model components not properly initialized")
        
        # Prepare the prompt (similar to text_demo.py)
        if system_message:
            prompt = f"{system_message}\n\n{user_message}"
        else:
            prompt = user_message
        
        # Tokenize the input
        input_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        input_tokens_tensor = torch.tensor([input_tokens])
        
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Request timed out after {timeout}s during preprocessing")
        
        # Perform inference (simplified version of text_demo.py inference)
        try:
            # Prefill
            model.switch_mode("prefill")
            prefill_output = generator.prefill_forward_text(
                input_tokens_tensor,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=[len(input_tokens)],
                enable_trace=False,
            )
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Request timed out after {timeout}s during prefill")
            
            # Decode
            model.switch_mode("decode")
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=REQUEST_PAYLOAD.get("temperature", 0.0),
                top_k=32,
                top_p=0.08
            )
            
            generated_tokens = []
            current_pos = torch.tensor([len(input_tokens)])
            out_tok = prefill_output.view(-1, 1)
            
            # Pad for batch size if needed
            if out_tok.shape[0] != 32:
                out_tok = out_tok.repeat(32, 1)
            if current_pos.shape[0] != 32:
                current_pos = torch.nn.functional.pad(current_pos, (0, 32 - current_pos.shape[0]), value=-1)
            if page_table is not None and page_table.shape[0] != 32:
                page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 32 - page_table.shape[0]), value=0)
            
            for i in range(max_generated_tokens):
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Request timed out after {timeout}s during decode iteration {i}")
                
                tt_out_tok, _ = generator.decode_forward_text(
                    out_tok,
                    current_pos,
                    enable_trace=False,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    read_from_device=True,
                    sampling_params=sampling_params,
                    reset_inputs=(i == 0),
                )
                
                # Process output
                out_tok = generator.process_decode_output_host(tt_out_tok)
                if out_tok.shape == torch.Size([]) or (len(out_tok.shape) > 0 and out_tok.shape[0] != 32):
                    out_tok = out_tok.repeat(32, 1)
                
                # Get the actual token for the first user
                token = int(out_tok[0].item())
                generated_tokens.append(token)
                
                # Check for stop tokens
                if token in tokenizer.stop_tokens:
                    break
                
                current_pos += 1
            
            # Decode the generated text
            full_tokens = input_tokens + generated_tokens
            generated_text = tokenizer.decode(generated_tokens)
            full_text = tokenizer.decode(full_tokens)
            
            processing_time = time.time() - start_time
            
            # Create response in OpenAI-like format
            response_data = {
                "id": f"local-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": REQUEST_PAYLOAD["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop" if token in tokenizer.stop_tokens else "length"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(input_tokens),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(full_tokens)
                }
            }
            
            result = RequestResult(
                request_id=request_id,
                success=True,
                response=response_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = RequestResult(
                request_id=request_id,
                success=False,
                error=f"Inference error: {str(e)}",
                processing_time=processing_time
            )
            
    except Exception as e:
        processing_time = time.time() - start_time
        result = RequestResult(
            request_id=request_id,
            success=False,
            error=str(e),
            processing_time=processing_time
        )
    
    # Thread-safe result storage and immediate JSON writing
    with lock:
        results_list.append(result)
        print(f"Request {request_id} completed in {result.processing_time:.2f}s - {'SUCCESS' if result.success else 'FAILED'}")
    
    # Write result to incremental JSON file immediately
    write_result_to_incremental_json(result, json_lock)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Concurrent Local LLM Inference Script - Makes concurrent requests to a locally loaded LLM model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--concurrency',
        type=int,
        default=32,
        help='Maximum number of concurrent requests (default: 32)'
    )
    
    parser.add_argument(
        '--loops',
        type=int,
        default=3,
        help='Number of times to loop over (concurrency) requests being sent to model (default: 3)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=None,
        help='Set timeout for each request in seconds (e.g., 0.01 for almost immediate timeout, default: no timeout)'
    )
    
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=128 * 1024,
        help='Maximum context length supported by the model (default: 128 * 1024)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of users in a batch (default: 1)'
    )
    
    parser.add_argument(
        '--max_generated_tokens',
        type=int,
        default=128,
        help='Maximum number of tokens to generate for each request (default: 128)'
    )
    
    parser.add_argument(
        '--mesh_device_config',
        type=str,
        default='(8,4)',
        help='Mesh device configuration as tuple string (default: "(8,4)")'
    )
    
    return parser.parse_args()

def cleanup_model():
    """Clean up model resources"""
    global model_components
    try:
        if model_components.get('model') and hasattr(model_components['model'], 'tt_ccl'):
            model_components['model'].tt_ccl.close()
        print("Model cleanup completed")
    except Exception as e:
        print(f"Error during model cleanup: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_model()
    close_json_array()
    sys.exit(0)

def main():
    global results_written_lock
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_requests = args.concurrency * args.loops
    print(f"Starting {total_requests} total requests ({args.concurrency} concurrent × {args.loops} loops) to local model")
    if args.timeout is not None:
        print(f"Request timeout: {args.timeout}s")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max generated tokens: {args.max_generated_tokens}")
    print(f"Results will be written incrementally to: {JSON_FILENAME}")
    print("=" * 60)
    
    # Initialize mesh device with device parameters (following the pattern from text_demo.py)
    try:
        # Parse mesh device config
        mesh_config = eval(args.mesh_device_config)  # e.g., (8, 4)
        print(f"Initializing mesh device with config: {mesh_config}")
        
        # Configure device parameters like text_demo.py (hardcoded)
        device_params = {
            "trace_region_size": 140280832,
            "num_command_queues": 1,
            "dispatch_core_axis": "col",
            "sample_on_device_mode": "all", 
            "fabric_config": "FABRIC_1D_RING",
            "worker_l1_size": 1344544,
        }
        
        print(f"Device parameters: {device_params}")
        
        # Follow the same pattern as the mesh_device fixture in conftest.py
        device_ids = ttnn.get_device_ids()
        
        if isinstance(mesh_config, tuple):
            grid_dims = mesh_config
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if not ttnn.using_distributed_env() and num_devices_requested > len(device_ids):
                print(f"Requested {num_devices_requested} devices but only {len(device_ids)} available")
                return 1
            mesh_shape = ttnn.MeshShape(*grid_dims)
        else:
            num_devices_requested = min(mesh_config, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)
        
        print(f"Opening mesh device with shape: {mesh_shape}")
        
        # Configure devices with parameters before opening mesh device
        for device_id in device_ids[:num_devices_requested]:
            ttnn.device.configure_device(
                device_id,
                trace_region_size=device_params["trace_region_size"],
                num_command_queues=device_params["num_command_queues"],
                dispatch_core_axis=ttnn.DispatchCoreAxis.COL,
                worker_l1_size=device_params["worker_l1_size"],
                fabric_config=True,
            )
        
        # Use the same approach as conftest.py mesh_device fixture
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
        
        print(f"Mesh device initialized successfully with {mesh_device.get_num_devices()} devices")
    except Exception as e:
        print(f"Failed to initialize mesh device: {e}")
        print("Make sure you're running in a proper tt-metal environment with TT devices available.")
        return 1
    
    # Initialize model components
    print("Initializing model components...")
    if not initialize_model_components(mesh_device, args.max_seq_len, args.max_generated_tokens, args.batch_size):
        print("Failed to initialize model components. Exiting.")
        return 1
    print("Model components initialized successfully")
    
    # Initialize global lock
    results_written_lock = threading.Lock()
    
    # Shared data structures
    results = []
    lock = threading.Lock()
    json_lock = threading.Lock()  # Separate lock for incremental JSON file writing
    
    try:
        # Start overall timing
        overall_start_time = time.time()
        
        # Loop over the specified number of loops
        for loop_num in range(args.loops):
            print(f"\n--- Loop {loop_num + 1}/{args.loops} ---")
            threads = []
            loop_start_time = time.time()
            
            # Start all threads for this loop
            for i in range(args.concurrency):
                request_id = str(uuid.uuid4())
                thread = threading.Thread(
                    target=make_local_inference_request,
                    args=(request_id, results, lock, json_lock, args.timeout, args.max_generated_tokens)
                )
                threads.append(thread)
                thread.start()
                print(f"Started request {i+1}/{args.concurrency} (ID: {request_id})")

            
            # Wait for all threads in this loop to complete
            for thread in threads:
                thread.join()
            
            loop_time = time.time() - loop_start_time
            print(f"Loop {loop_num + 1} completed in {loop_time:.2f} seconds")
        
        total_time = time.time() - overall_start_time
        
        print("=" * 60)
        print(f"All requests completed in {total_time:.2f} seconds")
        
        # Generate statistics
        successful_requests = [r for r in results if r.success]
        failed_requests = [r for r in results if not r.success]
        
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Failed requests: {len(failed_requests)}")
        
        if successful_requests:
            avg_time = sum(r.processing_time for r in successful_requests) / len(successful_requests)
            min_time = min(r.processing_time for r in successful_requests)
            max_time = max(r.processing_time for r in successful_requests)
            print(f"Average response time: {avg_time:.2f}s")
            print(f"Min response time: {min_time:.2f}s")
            print(f"Max response time: {max_time:.2f}s")
        
        print(f"Results saved to: {JSON_FILENAME}")
        
        # Print summary of failed requests if any
        if failed_requests:
            print("\nFailed requests summary:")
            for result in failed_requests:
                print(f"  {result.request_id}: {result.error}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Always clean up and close the incremental JSON array
        cleanup_model()
        close_json_array()
        # Close mesh device (following conftest.py pattern)
        try:
            if 'mesh_device' in locals():
                # Close submeshes first, then the main mesh device (like conftest.py)
                for submesh in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(mesh_device)
                print("Mesh device closed successfully")
        except Exception as e:
            print(f"Error closing mesh device: {e}")


def set_up_environment():
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
    sys.exit(main())
