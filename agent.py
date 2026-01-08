#!/usr/bin/env python3
"""
LLM-Assisted C++ Program Understanding Agent (Python Implementation)

Module 1: Primary Analysis Agent
Module 2: Description Validation Agent
Module 3: Diagram Validation Agent

HARD RULES:
- CFG is single source of truth
- Never infer control flow
- Never modify CFG or CallGraph
- Never add branches/loops not in CFG
- If information is missing → say "Not present in CFG"
"""

import json
import sys
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CFGNode:
    """Represents a node in the Control Flow Graph"""
    id: int
    type: str  # entry, condition, call, return, statement
    expr: str = ""  # for conditions
    callee: str = ""  # for calls
    label: str = ""  # optional node label


@dataclass
class CFGEdge:
    """Represents an edge in the Control Flow Graph"""
    from_node: int
    to_node: int
    label: str = ""  # "true", "false", or empty


@dataclass
class CFG:
    """Control Flow Graph structure"""
    function: str
    nodes: List[CFGNode] = field(default_factory=list)
    edges: List[CFGEdge] = field(default_factory=list)


@dataclass
class Description:
    """Semantic description metadata"""
    function: str
    summary: str = ""
    notes: str = ""
    validated: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class CallGraph:
    """Call graph structure"""
    calls: Dict[str, List[str]] = field(default_factory=dict)


# ============================================================================
# JSON PARSING
# ============================================================================

def parse_cfg(cfg_json: dict) -> CFG:
    """Parse CFG from JSON (READ-ONLY - never modify)"""
    cfg = CFG(function=cfg_json.get("function", ""))
    
    for node_data in cfg_json.get("nodes", []):
        node = CFGNode(
            id=node_data.get("id", -1),
            type=node_data.get("type", ""),
            expr=node_data.get("expr", ""),
            callee=node_data.get("callee", ""),
            label=node_data.get("label", "")
        )
        cfg.nodes.append(node)
    
    for edge_data in cfg_json.get("edges", []):
        edge = CFGEdge(
            from_node=edge_data.get("from", -1),
            to_node=edge_data.get("to", -1),
            label=edge_data.get("label", "")
        )
        cfg.edges.append(edge)
    
    return cfg


def parse_callgraph(callgraph_json: dict) -> CallGraph:
    """Parse CallGraph from JSON"""
    cg = CallGraph()
    
    if "functions" in callgraph_json:
        for func_name, callees in callgraph_json["functions"].items():
            if isinstance(callees, list):
                cg.calls[func_name] = [str(c) for c in callees if isinstance(c, str)]
    
    return cg


def parse_description(desc_json: dict) -> Description:
    """Parse Description from JSON"""
    desc = Description(function="", summary="", notes="")
    
    if desc_json and isinstance(desc_json, dict):
        # Assume first key is function name
        if desc_json:
            func_name = list(desc_json.keys())[0]
            func_desc = desc_json[func_name]
            
            desc.function = func_name
            desc.summary = func_desc.get("summary", "")
            desc.notes = func_desc.get("notes", "")
            desc.validated = func_desc.get("validated", False)
            desc.issues = func_desc.get("issues", [])
    
    return desc


# ============================================================================
# MODULE 2: DESCRIPTION VALIDATION AGENT
# ============================================================================

class DescriptionValidationAgent:
    """Validates Description JSON against CFG and CallGraph"""
    
    SPECULATIVE_WORDS = [
        "probably", "might", "seems", "appears", "likely", "possibly",
        "perhaps", "may", "could", "should", "would"
    ]
    
    @staticmethod
    def validate(desc: Description, cfg: CFG, callgraph: CallGraph) -> Tuple[bool, Description, str]:
        """
        Validate description against CFG and CallGraph.
        
        Returns:
            (is_valid, corrected_description, justification)
        """
        issues = []
        
        # Extract all condition expressions from CFG
        cfg_conditions = set()
        for node in cfg.nodes:
            if node.type == "condition" and node.expr:
                cfg_conditions.add(node.expr)
        
        # Extract all function calls from CFG
        cfg_calls = set()
        for node in cfg.nodes:
            if node.type == "call" and node.callee:
                cfg_calls.add(node.callee)
        
        # Extract all calls from CallGraph
        cg_calls = set()
        if cfg.function in callgraph.calls:
            cg_calls.update(callgraph.calls[cfg.function])
        
        # Combine all valid calls
        valid_calls = cfg_calls.union(cg_calls)
        
        # Check for speculative language
        combined_text = (desc.summary + " " + desc.notes).lower()
        for word in DescriptionValidationAgent.SPECULATIVE_WORDS:
            if word in combined_text:
                issues.append(f"Contains speculative language: '{word}'")
        
        # Validate function calls mentioned in description
        # Extract mentioned function names (simplified regex)
        call_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
        mentioned_calls = set(re.findall(call_pattern, desc.summary + " " + desc.notes))
        
        # Exclude common words that match function pattern
        excluded_words = {"is", "has", "was", "are", "get", "set"}
        for call in mentioned_calls:
            if call not in excluded_words and call not in valid_calls:
                issues.append(f"Mentions function call '{call}' not present in CFG or CallGraph")
        
        # If invalid, create corrected description
        if issues:
            corrected = Description(
                function=desc.function,
                summary=desc.summary,
                notes=desc.notes,
                validated=False,
                issues=issues
            )
            
            # Remove speculative language
            corrected_summary = desc.summary
            corrected_notes = desc.notes
            
            for word in DescriptionValidationAgent.SPECULATIVE_WORDS:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                corrected_summary = pattern.sub('', corrected_summary)
                corrected_notes = pattern.sub('', corrected_notes)
            
            # Clean up extra spaces
            corrected_summary = re.sub(r'\s+', ' ', corrected_summary).strip()
            corrected_notes = re.sub(r'\s+', ' ', corrected_notes).strip()
            
            corrected.summary = corrected_summary
            corrected.notes = corrected_notes
            
            justification = "Description invalid. Issues: " + "; ".join(issues)
            return False, corrected, justification
        else:
            corrected = Description(
                function=desc.function,
                summary=desc.summary,
                notes=desc.notes,
                validated=True,
                issues=[]
            )
            return True, corrected, "Description is valid and aligned with CFG/CallGraph"


# ============================================================================
# MODULE 3: DIAGRAM VALIDATION AGENT
# ============================================================================

class DiagramValidationAgent:
    """Validates Mermaid diagrams against CFG"""
    
    @staticmethod
    def validate(mermaid: str, cfg: CFG) -> Tuple[bool, str, List[str]]:
        """
        Validate Mermaid diagram against CFG.
        
        Returns:
            (is_valid, corrected_mermaid, issues)
        """
        issues = []
        
        # Extract nodes from Mermaid
        node_pattern = r'(\d+)\['
        mermaid_nodes = set(int(m.group(1)) for m in re.finditer(node_pattern, mermaid))
        
        # Extract edges from Mermaid (handle labels like |"label"|)
        edge_pattern = r'(\d+)-->(?:\|"[^"]*"\||)(\d+)'
        mermaid_edges = set()
        for m in re.finditer(edge_pattern, mermaid):
            from_node = int(m.group(1))
            to_node = int(m.group(2))
            mermaid_edges.add((from_node, to_node))
        
        # Extract CFG nodes
        cfg_nodes = {node.id for node in cfg.nodes}
        
        # Extract CFG edges
        cfg_edges = {(edge.from_node, edge.to_node) for edge in cfg.edges}
        
        # Validate node count
        if len(mermaid_nodes) != len(cfg_nodes):
            issues.append(f"Node count mismatch: Mermaid has {len(mermaid_nodes)}, CFG has {len(cfg_nodes)}")
        
        # Check for missing nodes
        missing_nodes = cfg_nodes - mermaid_nodes
        for node_id in missing_nodes:
            issues.append(f"Missing CFG node: {node_id}")
        
        # Check for extra nodes
        extra_nodes = mermaid_nodes - cfg_nodes
        for node_id in extra_nodes:
            issues.append(f"Extra node in diagram: {node_id} (not in CFG)")
        
        # Validate edge count
        if len(mermaid_edges) != len(cfg_edges):
            issues.append(f"Edge count mismatch: Mermaid has {len(mermaid_edges)}, CFG has {len(cfg_edges)}")
        
        # Check for missing edges
        missing_edges = cfg_edges - mermaid_edges
        for edge in missing_edges:
            issues.append(f"Missing CFG edge: {edge[0]} -> {edge[1]}")
        
        # Check for extra edges
        extra_edges = mermaid_edges - cfg_edges
        for edge in extra_edges:
            issues.append(f"Extra edge in diagram: {edge[0]} -> {edge[1]} (not in CFG)")
        
        # If invalid, generate corrected Mermaid
        if issues:
            corrected = DiagramValidationAgent._generate_corrected_mermaid(cfg)
            return False, corrected, issues
        else:
            return True, mermaid, []
    
    @staticmethod
    def _generate_corrected_mermaid(cfg: CFG) -> str:
        """Generate corrected Mermaid from CFG (CFG is source of truth)"""
        lines = ["flowchart TD"]
        
        # Add nodes from CFG
        for node in cfg.nodes:
            node_label = ""
            if node.label:
                node_label = node.label
            elif node.type == "entry":
                node_label = f"Entry: {cfg.function}"
            elif node.type == "return":
                node_label = "Return"
            elif node.type == "condition":
                node_label = node.expr
            elif node.type == "call":
                node_label = f"Call: {node.callee}"
            else:
                node_label = f"Node {node.id}"
            
            lines.append(f"    {node.id}[\"{node_label}\"]")
        
        # Add edges from CFG (NEVER add edges not in CFG)
        for edge in cfg.edges:
            edge_str = f"    {edge.from_node}-->"
            if edge.label:
                edge_str += f"|\"{edge.label}\"|"
            edge_str += f"{edge.to_node}"
            lines.append(edge_str)
        
        return "\n".join(lines)


# ============================================================================
# MODULE 1: PRIMARY ANALYSIS AGENT
# ============================================================================

class PrimaryAnalysisAgent:
    """Primary agent that orchestrates analysis and diagram generation"""
    
    @staticmethod
    def generate_mermaid(cfg: CFG, desc: Description) -> str:
        """
        Generate Mermaid diagram from CFG.
        Uses CFG for structure, Description only for labels.
        """
        lines = ["flowchart TD"]
        
        # Generate nodes from CFG (CFG is source of truth)
        for node in cfg.nodes:
            # Use description for labels if available, otherwise use CFG info
            node_label = ""
            if node.type == "entry":
                node_label = f"Entry: {cfg.function}"
            elif node.type == "return":
                node_label = "Return"
            elif node.type == "condition":
                node_label = node.expr
            elif node.type == "call":
                node_label = f"Call: {node.callee}"
            else:
                node_label = f"Node {node.id}"
            
            lines.append(f"    {node.id}[\"{node_label}\"]")
        
        # Generate edges from CFG (NEVER add edges not in CFG)
        for edge in cfg.edges:
            edge_str = f"    {edge.from_node}-->"
            if edge.label:
                edge_str += f"|\"{edge.label}\"|"
            edge_str += f"{edge.to_node}"
            lines.append(edge_str)
        
        return "\n".join(lines)


# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

def main():
    """Main execution flow - follows exact sequence"""
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cfg_json_file> <description_json_file> [callgraph_json_file]")
        sys.exit(1)
    
    cfg_file = sys.argv[1]
    desc_file = sys.argv[2]
    callgraph_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Step 1: Load CFG JSON
    try:
        with open(cfg_file, 'r') as f:
            cfg_json = json.load(f)
        cfg = parse_cfg(cfg_json)
    except FileNotFoundError:
        print(f"Error: Cannot open CFG file: {cfg_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in CFG file: {e}")
        sys.exit(1)
    
    if not cfg.nodes:
        print("Warning: CFG has no nodes. Aborting.")
        sys.exit(1)
    
    print(f"[OK] Loaded CFG for function: {cfg.function}")
    print(f"  Nodes: {len(cfg.nodes)}, Edges: {len(cfg.edges)}")
    
    # Step 2: Load Description JSON
    try:
        with open(desc_file, 'r') as f:
            desc_json = json.load(f)
        desc = parse_description(desc_json)
    except FileNotFoundError:
        print(f"Error: Cannot open Description file: {desc_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in Description file: {e}")
        sys.exit(1)
    
    print(f"[OK] Loaded Description for: {desc.function}")
    
    # Step 3: Load CallGraph JSON (optional)
    callgraph = CallGraph()
    if callgraph_file:
        try:
            with open(callgraph_file, 'r') as f:
                cg_json = json.load(f)
            callgraph = parse_callgraph(cg_json)
            print("[OK] Loaded CallGraph")
        except FileNotFoundError:
            print(f"Warning: Cannot open CallGraph file: {callgraph_file}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in CallGraph file: {e}")
    
    # Step 4: Run Description Validation Agent
    desc_validator = DescriptionValidationAgent()
    is_valid, corrected_desc, justification = desc_validator.validate(desc, cfg, callgraph)
    
    if not is_valid:
        print("\n[WARNING] Description Validation FAILED")
        print(f"  {justification}")
        
        # Overwrite description JSON with corrected version
        corrected_json = {
            corrected_desc.function: {
                "summary": corrected_desc.summary,
                "notes": corrected_desc.notes,
                "validated": corrected_desc.validated,
                "issues": corrected_desc.issues
            }
        }
        
        try:
            with open(desc_file, 'w') as f:
                json.dump(corrected_json, f, indent=2)
            print(f"  [OK] Corrected description saved to: {desc_file}")
        except Exception as e:
            print(f"  Error saving corrected description: {e}")
        
        # Update desc for further use
        desc = corrected_desc
    else:
        print("\n[OK] Description Validation PASSED")
        print(f"  {justification}")
    
    # Step 5: Generate Mermaid diagram
    primary_agent = PrimaryAnalysisAgent()
    mermaid = primary_agent.generate_mermaid(cfg, desc)
    
    print("\n[OK] Generated Mermaid diagram")
    
    # Step 6: Run Diagram Validation Agent
    diagram_validator = DiagramValidationAgent()
    is_valid, corrected_mermaid, issues = diagram_validator.validate(mermaid, cfg)
    
    validation_attempts = 1
    while not is_valid and validation_attempts < 3:
        print(f"\n[WARNING] Diagram Validation FAILED (attempt {validation_attempts})")
        for issue in issues:
            print(f"  - {issue}")
        
        mermaid = corrected_mermaid
        is_valid, corrected_mermaid, issues = diagram_validator.validate(mermaid, cfg)
        validation_attempts += 1
    
    if is_valid:
        print("\n[OK] Diagram Validation PASSED")
    else:
        print(f"\n[ERROR] Diagram Validation FAILED after {validation_attempts} attempts")
        print("  Aborting due to persistent validation failures.")
        sys.exit(1)
    
    # Step 7: Output final Mermaid
    print("\n=== FINAL MERMAID DIAGRAM ===\n")
    print(mermaid)
    
    # Save to file
    try:
        with open("output.mermaid", 'w') as f:
            f.write(mermaid + "\n")
        print("\n[OK] Saved to: output.mermaid")
    except Exception as e:
        print(f"\nWarning: Could not save to output.mermaid: {e}")
    
    # Step 8: Short explanation
    print("\n=== EXPLANATION ===")
    print(f"Function: {cfg.function}")
    print(f"Summary: {desc.summary}")
    print(f"Flow: {len(cfg.nodes)} nodes, {len(cfg.edges)} edges")
    if desc.notes:
        print(f"Notes: {desc.notes}")


if __name__ == "__main__":
    main()

