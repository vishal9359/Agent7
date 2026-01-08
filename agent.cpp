/**
 * LLM-Assisted C++ Program Understanding Agent
 * 
 * Module 1: Primary Analysis Agent
 * Module 2: Description Validation Agent
 * Module 3: Diagram Validation Agent
 * 
 * HARD RULES:
 * - CFG is single source of truth
 * - Never infer control flow
 * - Never modify CFG or CallGraph
 * - Never add branches/loops not in CFG
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include <regex>

// JSON parsing (using nlohmann/json - include via CMake)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ============================================================================
// DATA MODELS
// ============================================================================

struct CFGNode {
    int id;
    std::string type;  // entry, condition, call, return, statement
    std::string expr;  // for conditions
    std::string callee; // for calls
    std::string label; // optional node label
};

struct CFGEdge {
    int from;
    int to;
    std::string label; // "true", "false", or empty
};

struct CFG {
    std::string function;
    std::vector<CFGNode> nodes;
    std::vector<CFGEdge> edges;
};

struct CallGraph {
    std::map<std::string, std::vector<std::string>> calls; // function -> [callees]
};

struct Description {
    std::string function;
    std::string summary;
    std::string notes;
    bool validated = false;
    std::vector<std::string> issues;
};

// ============================================================================
// JSON PARSING
// ============================================================================

CFG parseCFG(const json& cfgJson) {
    CFG cfg;
    cfg.function = cfgJson.value("function", "");
    
    if (cfgJson.contains("nodes")) {
        for (const auto& nodeJson : cfgJson["nodes"]) {
            CFGNode node;
            node.id = nodeJson.value("id", -1);
            node.type = nodeJson.value("type", "");
            node.expr = nodeJson.value("expr", "");
            node.callee = nodeJson.value("callee", "");
            node.label = nodeJson.value("label", "");
            cfg.nodes.push_back(node);
        }
    }
    
    if (cfgJson.contains("edges")) {
        for (const auto& edgeJson : cfgJson["edges"]) {
            CFGEdge edge;
            edge.from = edgeJson.value("from", -1);
            edge.to = edgeJson.value("to", -1);
            edge.label = edgeJson.value("label", "");
            cfg.edges.push_back(edge);
        }
    }
    
    return cfg;
}

CallGraph parseCallGraph(const json& callGraphJson) {
    CallGraph cg;
    if (callGraphJson.contains("functions")) {
        for (const auto& func : callGraphJson["functions"].items()) {
            std::string funcName = func.key();
            std::vector<std::string> callees;
            if (func.value().is_array()) {
                for (const auto& callee : func.value()) {
                    if (callee.is_string()) {
                        callees.push_back(callee);
                    }
                }
            }
            cg.calls[funcName] = callees;
        }
    }
    return cg;
}

Description parseDescription(const json& descJson) {
    Description desc;
    if (descJson.is_object() && !descJson.empty()) {
        // Assume first key is function name
        auto it = descJson.begin();
        desc.function = it.key();
        const auto& funcDesc = it.value();
        desc.summary = funcDesc.value("summary", "");
        desc.notes = funcDesc.value("notes", "");
        desc.validated = funcDesc.value("validated", false);
        if (funcDesc.contains("issues")) {
            for (const auto& issue : funcDesc["issues"]) {
                if (issue.is_string()) {
                    desc.issues.push_back(issue);
                }
            }
        }
    }
    return desc;
}

// ============================================================================
// MODULE 2: DESCRIPTION VALIDATION AGENT
// ============================================================================

class DescriptionValidationAgent {
public:
    struct ValidationResult {
        bool valid;
        Description corrected;
        std::string justification;
    };
    
    ValidationResult validate(const Description& desc, const CFG& cfg, const CallGraph& callGraph) {
        ValidationResult result;
        result.valid = true;
        result.corrected = desc;
        std::vector<std::string> issues;
        
        // Extract all condition expressions from CFG
        std::set<std::string> cfgConditions;
        for (const auto& node : cfg.nodes) {
            if (node.type == "condition" && !node.expr.empty()) {
                cfgConditions.insert(node.expr);
            }
        }
        
        // Extract all function calls from CFG
        std::set<std::string> cfgCalls;
        for (const auto& node : cfg.nodes) {
            if (node.type == "call" && !node.callee.empty()) {
                cfgCalls.insert(node.callee);
            }
        }
        
        // Extract all calls from CallGraph
        std::set<std::string> cgCalls;
        if (callGraph.calls.find(cfg.function) != callGraph.calls.end()) {
            for (const auto& callee : callGraph.calls.at(cfg.function)) {
                cgCalls.insert(callee);
            }
        }
        
        // Combine all valid calls
        std::set<std::string> validCalls;
        validCalls.insert(cfgCalls.begin(), cfgCalls.end());
        validCalls.insert(cgCalls.begin(), cgCalls.end());
        
        // Check for speculative language
        std::vector<std::string> speculativeWords = {
            "probably", "might", "seems", "appears", "likely", "possibly",
            "perhaps", "may", "could", "should", "would"
        };
        
        std::string combinedText = desc.summary + " " + desc.notes;
        std::transform(combinedText.begin(), combinedText.end(), combinedText.begin(), ::tolower);
        
        for (const auto& word : speculativeWords) {
            if (combinedText.find(word) != std::string::npos) {
                issues.push_back("Contains speculative language: '" + word + "'");
                result.valid = false;
            }
        }
        
        // Validate against CFG conditions (basic check)
        // Check if description mentions conditions not in CFG
        // This is simplified - in production, use NLP for better matching
        
        // Validate function calls
        // Extract mentioned function names from description (simplified)
        std::regex callRegex(R"(\b([A-Za-z_][A-Za-z0-9_]*)\s*\()");
        std::smatch matches;
        std::string textToCheck = desc.summary + " " + desc.notes;
        
        std::set<std::string> mentionedCalls;
        std::string::const_iterator searchStart(textToCheck.cbegin());
        while (std::regex_search(searchStart, textToCheck.cend(), matches, callRegex)) {
            mentionedCalls.insert(matches[1].str());
            searchStart = matches.suffix().first;
        }
        
        // Check if any mentioned calls are not in valid calls
        for (const auto& call : mentionedCalls) {
            // Exclude common words that match function pattern
            if (call != "is" && call != "has" && call != "was" && call != "are") {
                if (validCalls.find(call) == validCalls.end() && !call.empty()) {
                    issues.push_back("Mentions function call '" + call + "' not present in CFG or CallGraph");
                    result.valid = false;
                }
            }
        }
        
        // If invalid, create corrected description
        if (!result.valid) {
            result.corrected.issues = issues;
            result.corrected.validated = false;
            
            // Remove speculative language
            std::string correctedSummary = desc.summary;
            std::string correctedNotes = desc.notes;
            
            for (const auto& word : speculativeWords) {
                std::regex wordRegex("\\b" + word + "\\b", std::regex_constants::icase);
                correctedSummary = std::regex_replace(correctedSummary, wordRegex, "");
                correctedNotes = std::regex_replace(correctedNotes, wordRegex, "");
            }
            
            // Clean up extra spaces
            correctedSummary = std::regex_replace(correctedSummary, std::regex("\\s+"), " ");
            correctedNotes = std::regex_replace(correctedNotes, std::regex("\\s+"), " ");
            
            result.corrected.summary = correctedSummary;
            result.corrected.notes = correctedNotes;
            
            std::ostringstream oss;
            oss << "Description invalid. Issues: ";
            for (size_t i = 0; i < issues.size(); i++) {
                oss << issues[i];
                if (i < issues.size() - 1) oss << "; ";
            }
            result.justification = oss.str();
        } else {
            result.corrected.validated = true;
            result.corrected.issues.clear();
            result.justification = "Description is valid and aligned with CFG/CallGraph";
        }
        
        return result;
    }
};

// ============================================================================
// MODULE 3: DIAGRAM VALIDATION AGENT
// ============================================================================

class DiagramValidationAgent {
public:
    struct ValidationResult {
        bool valid;
        std::string correctedMermaid;
        std::vector<std::string> issues;
    };
    
    ValidationResult validate(const std::string& mermaid, const CFG& cfg) {
        ValidationResult result;
        result.valid = true;
        result.correctedMermaid = mermaid;
        
        // Count nodes in Mermaid (simplified - count node definitions)
        std::regex nodeRegex(R"((\d+)\[)");
        std::smatch matches;
        std::set<int> mermaidNodes;
        
        std::string::const_iterator searchStart(mermaid.cbegin());
        while (std::regex_search(searchStart, mermaid.cend(), matches, nodeRegex)) {
            int nodeId = std::stoi(matches[1].str());
            mermaidNodes.insert(nodeId);
            searchStart = matches.suffix().first;
        }
        
        // Count edges in Mermaid
        std::regex edgeRegex(R"((\d+)-->(\d+))");
        std::set<std::pair<int, int>> mermaidEdges;
        searchStart = mermaid.cbegin();
        while (std::regex_search(searchStart, mermaid.cend(), matches, edgeRegex)) {
            int from = std::stoi(matches[1].str());
            int to = std::stoi(matches[2].str());
            mermaidEdges.insert({from, to});
            searchStart = matches.suffix().first;
        }
        
        // Count CFG nodes
        std::set<int> cfgNodes;
        for (const auto& node : cfg.nodes) {
            cfgNodes.insert(node.id);
        }
        
        // Count CFG edges
        std::set<std::pair<int, int>> cfgEdges;
        std::map<std::pair<int, int>, std::string> cfgEdgeLabels;
        for (const auto& edge : cfg.edges) {
            cfgEdges.insert({edge.from, edge.to});
            if (!edge.label.empty()) {
                cfgEdgeLabels[{edge.from, edge.to}] = edge.label;
            }
        }
        
        // Validate node count
        if (mermaidNodes.size() != cfgNodes.size()) {
            result.valid = false;
            result.issues.push_back("Node count mismatch: Mermaid has " + 
                                   std::to_string(mermaidNodes.size()) + 
                                   ", CFG has " + std::to_string(cfgNodes.size()));
        }
        
        // Check for missing nodes
        for (const auto& cfgNode : cfgNodes) {
            if (mermaidNodes.find(cfgNode) == mermaidNodes.end()) {
                result.valid = false;
                result.issues.push_back("Missing CFG node: " + std::to_string(cfgNode));
            }
        }
        
        // Check for extra nodes
        for (const auto& mNode : mermaidNodes) {
            if (cfgNodes.find(mNode) == cfgNodes.end()) {
                result.valid = false;
                result.issues.push_back("Extra node in diagram: " + std::to_string(mNode) + " (not in CFG)");
            }
        }
        
        // Validate edge count
        if (mermaidEdges.size() != cfgEdges.size()) {
            result.valid = false;
            result.issues.push_back("Edge count mismatch: Mermaid has " + 
                                   std::to_string(mermaidEdges.size()) + 
                                   ", CFG has " + std::to_string(cfgEdges.size()));
        }
        
        // Check for missing edges
        for (const auto& cfgEdge : cfgEdges) {
            if (mermaidEdges.find(cfgEdge) == mermaidEdges.end()) {
                result.valid = false;
                result.issues.push_back("Missing CFG edge: " + 
                                       std::to_string(cfgEdge.first) + " -> " + 
                                       std::to_string(cfgEdge.second));
            }
        }
        
        // Check for extra edges
        for (const auto& mEdge : mermaidEdges) {
            if (cfgEdges.find(mEdge) == cfgEdges.end()) {
                result.valid = false;
                result.issues.push_back("Extra edge in diagram: " + 
                                       std::to_string(mEdge.first) + " -> " + 
                                       std::to_string(mEdge.second) + 
                                       " (not in CFG)");
            }
        }
        
        // If invalid, regenerate (simplified - would need proper correction logic)
        if (!result.valid) {
            result.correctedMermaid = generateCorrectedMermaid(cfg);
        }
        
        return result;
    }
    
private:
    std::string generateCorrectedMermaid(const CFG& cfg) {
        // Generate corrected Mermaid from CFG
        std::ostringstream oss;
        oss << "flowchart TD\n";
        
        // Add nodes
        for (const auto& node : cfg.nodes) {
            oss << "    " << node.id << "[";
            
            if (!node.label.empty()) {
                oss << node.label;
            } else if (node.type == "entry") {
                oss << "Entry";
            } else if (node.type == "return") {
                oss << "Return";
            } else if (node.type == "condition") {
                oss << node.expr;
            } else if (node.type == "call") {
                oss << "Call: " << node.callee;
            } else {
                oss << "Node " << node.id;
            }
            
            oss << "]\n";
        }
        
        // Add edges
        for (const auto& edge : cfg.edges) {
            oss << "    " << edge.from << "-->";
            
            if (!edge.label.empty()) {
                oss << "|" << edge.label << "|";
            }
            
            oss << edge.to << "\n";
        }
        
        return oss.str();
    }
};

// ============================================================================
// MODULE 1: PRIMARY ANALYSIS AGENT
// ============================================================================

class PrimaryAnalysisAgent {
public:
    std::string generateMermaid(const CFG& cfg, const Description& desc) {
        std::ostringstream oss;
        oss << "flowchart TD\n";
        
        // Generate nodes from CFG
        for (const auto& node : cfg.nodes) {
            oss << "    " << node.id << "[";
            
            // Use description for labels if available, otherwise use CFG info
            std::string nodeLabel;
            if (node.type == "entry") {
                nodeLabel = "Entry: " + cfg.function;
            } else if (node.type == "return") {
                nodeLabel = "Return";
            } else if (node.type == "condition") {
                nodeLabel = node.expr;
            } else if (node.type == "call") {
                nodeLabel = "Call: " + node.callee;
            } else {
                nodeLabel = "Node " + std::to_string(node.id);
            }
            
            // Use description notes for context if available
            if (node.type == "call" && !desc.notes.empty()) {
                // Could add description context here, but keep it minimal
            }
            
            oss << nodeLabel << "]\n";
        }
        
        // Generate edges from CFG (NEVER add edges not in CFG)
        for (const auto& edge : cfg.edges) {
            oss << "    " << edge.from << "-->";
            
            if (!edge.label.empty()) {
                oss << "|" << edge.label << "|";
            }
            
            oss << edge.to << "\n";
        }
        
        return oss.str();
    }
};

// ============================================================================
// MAIN EXECUTION FLOW
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <cfg_json_file> <description_json_file> [callgraph_json_file]\n";
        return 1;
    }
    
    std::string cfgFile = argv[1];
    std::string descFile = argv[2];
    std::string callGraphFile = (argc > 3) ? argv[3] : "";
    
    // Step 1: Load CFG JSON
    std::ifstream cfgStream(cfgFile);
    if (!cfgStream.is_open()) {
        std::cerr << "Error: Cannot open CFG file: " << cfgFile << "\n";
        return 1;
    }
    
    json cfgJson;
    cfgStream >> cfgJson;
    CFG cfg = parseCFG(cfgJson);
    
    if (cfg.nodes.empty()) {
        std::cerr << "Warning: CFG has no nodes. Aborting.\n";
        return 1;
    }
    
    std::cout << "✓ Loaded CFG for function: " << cfg.function << "\n";
    std::cout << "  Nodes: " << cfg.nodes.size() << ", Edges: " << cfg.edges.size() << "\n";
    
    // Step 2: Load Description JSON
    std::ifstream descStream(descFile);
    if (!descStream.is_open()) {
        std::cerr << "Error: Cannot open Description file: " << descFile << "\n";
        return 1;
    }
    
    json descJson;
    descStream >> descJson;
    Description desc = parseDescription(descJson);
    
    std::cout << "✓ Loaded Description for: " << desc.function << "\n";
    
    // Step 3: Load CallGraph JSON (optional)
    CallGraph callGraph;
    if (!callGraphFile.empty()) {
        std::ifstream cgStream(callGraphFile);
        if (cgStream.is_open()) {
            json cgJson;
            cgStream >> cgJson;
            callGraph = parseCallGraph(cgJson);
            std::cout << "✓ Loaded CallGraph\n";
        } else {
            std::cerr << "Warning: Cannot open CallGraph file: " << callGraphFile << "\n";
        }
    }
    
    // Step 4: Run Description Validation Agent
    DescriptionValidationAgent descValidator;
    auto descResult = descValidator.validate(desc, cfg, callGraph);
    
    if (!descResult.valid) {
        std::cout << "\n⚠ Description Validation FAILED\n";
        std::cout << "  " << descResult.justification << "\n";
        
        // Overwrite description JSON with corrected version
        json correctedJson;
        correctedJson[descResult.corrected.function]["summary"] = descResult.corrected.summary;
        correctedJson[descResult.corrected.function]["notes"] = descResult.corrected.notes;
        correctedJson[descResult.corrected.function]["validated"] = descResult.corrected.validated;
        correctedJson[descResult.corrected.function]["issues"] = descResult.corrected.issues;
        
        std::ofstream descOut(descFile);
        descOut << correctedJson.dump(2) << "\n";
        std::cout << "  ✓ Corrected description saved to: " << descFile << "\n";
        
        // Update desc for further use
        desc = descResult.corrected;
    } else {
        std::cout << "\n✓ Description Validation PASSED\n";
        std::cout << "  " << descResult.justification << "\n";
    }
    
    // Step 5: Generate Mermaid diagram
    PrimaryAnalysisAgent primaryAgent;
    std::string mermaid = primaryAgent.generateMermaid(cfg, desc);
    
    std::cout << "\n✓ Generated Mermaid diagram\n";
    
    // Step 6: Run Diagram Validation Agent
    DiagramValidationAgent diagramValidator;
    auto diagramResult = diagramValidator.validate(mermaid, cfg);
    
    int validationAttempts = 1;
    while (!diagramResult.valid && validationAttempts < 3) {
        std::cout << "\n⚠ Diagram Validation FAILED (attempt " << validationAttempts << ")\n";
        for (const auto& issue : diagramResult.issues) {
            std::cout << "  - " << issue << "\n";
        }
        
        mermaid = diagramResult.correctedMermaid;
        diagramResult = diagramValidator.validate(mermaid, cfg);
        validationAttempts++;
    }
    
    if (diagramResult.valid) {
        std::cout << "\n✓ Diagram Validation PASSED\n";
    } else {
        std::cerr << "\n✗ Diagram Validation FAILED after " << validationAttempts << " attempts\n";
        std::cerr << "  Aborting due to persistent validation failures.\n";
        return 1;
    }
    
    // Step 7: Output final Mermaid
    std::cout << "\n=== FINAL MERMAID DIAGRAM ===\n\n";
    std::cout << mermaid << "\n";
    
    // Save to file
    std::ofstream mermaidOut("output.mermaid");
    mermaidOut << mermaid << "\n";
    std::cout << "\n✓ Saved to: output.mermaid\n";
    
    // Step 8: Short explanation
    std::cout << "\n=== EXPLANATION ===\n";
    std::cout << "Function: " << cfg.function << "\n";
    std::cout << "Summary: " << desc.summary << "\n";
    std::cout << "Flow: " << cfg.nodes.size() << " nodes, " << cfg.edges.size() << " edges\n";
    if (!desc.notes.empty()) {
        std::cout << "Notes: " << desc.notes << "\n";
    }
    
    return 0;
}

