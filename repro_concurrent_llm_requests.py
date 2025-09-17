#!/usr/bin/env python3
"""
Concurrent LLM API Request Script
Makes configurable concurrent requests to a local LLM API and saves results to a timestamped JSON file.
Supports multiple loops of concurrent requests with configurable timeout.
Uses only Python standard library.

CLI Options:
  --concurrency: Maximum number of concurrent requests (default: 32)
  --loops: Number of times to loop over (concurrency) requests (default: 3)
  --timeout: Timeout for each request in seconds (default: no timeout)
  --skip-trace-capture: Skip the warmup trace capture request (default: false, runs 1 warmup request)
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import threading
import time
from datetime import datetime
import uuid
import sys
import os
import argparse

# Configuration
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
API_TOKEN = os.getenv("API_TOKEN")

# Generate timestamped filename at startup
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = "output"
JSON_FILENAME = os.path.join(OUTPUT_DIR, f"concurrent_llm_requests_{TIMESTAMP}.json")

# Global counter for tracking written results (initialized after imports)
results_written_count = 0
results_written_lock = None

# Sample request payload
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

def make_api_request(request_id, results_list, lock, json_lock, timeout=None):
    """Make a single API request and store the result"""
    start_time = time.time()
    
    try:
        # Prepare the request
        data = json.dumps(REQUEST_PAYLOAD).encode('utf-8')
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if API_TOKEN:
            headers['Authorization'] = f'Bearer {API_TOKEN}'
        
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers=headers
        )
        
        # Make the request with optional timeout
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            processing_time = time.time() - start_time
            
            result = RequestResult(
                request_id=request_id,
                success=True,
                response=response_data,
                processing_time=processing_time
            )
            
    except urllib.error.HTTPError as e:
        processing_time = time.time() - start_time
        error_msg = f"HTTP {e.code}: {e.reason}"
        try:
            error_body = e.read().decode('utf-8')
            error_msg += f" - {error_body}"
        except:
            pass
            
        result = RequestResult(
            request_id=request_id,
            success=False,
            error=error_msg,
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
    
    # Thread-safe result storage and immediate JSONL writing
    with lock:
        results_list.append(result)
        print(f"Request {request_id} completed in {result.processing_time:.2f}s - {'SUCCESS' if result.success else 'FAILED'}")
    
    # Write result to incremental JSON file immediately
    write_result_to_incremental_json(result, json_lock)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Concurrent LLM API Request Script - Makes concurrent requests to a local LLM API",
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
        help='Number of times to loop over (concurrency) requests being sent to server (default: 3)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=None,
        help='Set timeout for each request in seconds (e.g., 0.01 for almost immediate timeout, default: no timeout)'
    )
    
    parser.add_argument(
        '--skip-trace-capture',
        action='store_true',
        default=False,
        help='Skip the warmup trace capture request (default: false, runs 1 warmup request)'
    )
    
    return parser.parse_args()

def main():
    global results_written_lock
    
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_requests = args.concurrency * args.loops
    print(f"Starting {total_requests} total requests ({args.concurrency} concurrent × {args.loops} loops) to {API_URL}")
    if args.timeout is not None:
        print(f"Request timeout: {args.timeout}s")
    print(f"Results will be written incrementally to: {JSON_FILENAME}")
    print("=" * 60)
    
    # Initialize global lock
    results_written_lock = threading.Lock()
    
    # Shared data structures
    results = []
    lock = threading.Lock()
    json_lock = threading.Lock()  # Separate lock for incremental JSON file writing
    
    try:
        # Run warmup trace capture request unless skipped
        if not args.skip_trace_capture:
            print("\n--- Warmup Trace Capture Request ---")
            warmup_start_time = time.time()
            
            # Create a separate results list for warmup (not included in main results)
            warmup_results = []
            warmup_request_id = str(uuid.uuid4())
            
            print(f"Starting warmup request (ID: {warmup_request_id})")
            
            # Run single warmup request
            TRACE_CAPTURE_TIMEOUT = 7200
            warmup_thread = threading.Thread(
                target=make_api_request,
                args=(warmup_request_id, warmup_results, lock, json_lock, TRACE_CAPTURE_TIMEOUT)
            )
            warmup_thread.start()
            warmup_thread.join()
            
            warmup_time = time.time() - warmup_start_time
            print(f"Warmup request completed in {warmup_time:.2f} seconds")
            
            if warmup_results and warmup_results[0].success:
                print("Warmup request successful - proceeding with main test")
            else:
                print("Warmup request failed - proceeding with main test anyway")
            
            print("=" * 60)
        
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
                    target=make_api_request,
                    args=(request_id, results, lock, json_lock, args.timeout)
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
        # Always close the incremental JSON array, even if script crashes
        close_json_array()

if __name__ == "__main__":
    sys.exit(main())
