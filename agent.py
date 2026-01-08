#!/usr/bin/env python3
"""
LLM-Assisted C++ Program Understanding Agent (Python Implementation)

Module 0: CFG Builder
Module 0.1: CallGraph Builder
Module 0.2: Description Builder
Module 1: Primary Analysis Agent
Module 2: Description Validation Agent
Module 3: Diagram Validation Agent
Module 4: CFG Validation Agent (NEW)
Module 5: CallGraph Validation Agent (NEW)

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
import os
import glob
import argparse
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
    
    def _fix_orphaned_nodes(self, entry_id: int):
        """Connect orphaned nodes to entry or return"""
        # Get all node IDs
        node_ids = {node.id for node in self.nodes}
        
        # Get nodes with incoming edges
        nodes_with_incoming = {edge.to_node for edge in self.edges}
        
        # Find orphaned nodes (except entry)
        orphaned = node_ids - nodes_with_incoming - {entry_id}
        
        # Connect orphaned nodes to entry
        for orphan_id in orphaned:
            # Check if this orphan already has an edge from entry
            has_entry_edge = any(e.from_node == entry_id and e.to_node == orphan_id for e in self.edges)
            if not has_entry_edge:
                self.edges.append(CFGEdge(from_node=entry_id, to_node=orphan_id))


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
# MODULE 0: CFG BUILDER
# ============================================================================

class CFGBuilder:
    """Builds CFG JSON from C++ source code"""
    
    @staticmethod
    def _read_cpp_file(filepath: str) -> str:
        """Read C++ file content, handling encoding issues"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"  Warning: Could not read {filepath}: {e}")
            return ""
    
    @staticmethod
    def _preprocess_code(code: str) -> str:
        """Remove comments and normalize whitespace"""
        # Remove single-line comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        return code
    
    @staticmethod
    def _find_functions(code: str) -> List[Dict]:
        """Extract function definitions from C++ code"""
        functions = []
        
        # More flexible pattern to match function definitions
        # Handles: return_type function_name(params) { ... }
        # Also handles: Class::method, namespace::function, etc.
        # Pattern breakdown:
        # 1. Return type (can include spaces, ::, <templates>, *, &)
        # 2. Function name (identifier or Class::method)
        # 3. Parameters
        # 4. Opening brace
        
        # Match function signatures more flexibly
        # This pattern handles: void func(), int Class::method(), etc.
        pattern = r'(?:[\w:]+\s+)+(\w+(?:::\w+)?)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(pattern, code):
            func_name = match.group(1).strip()
            
            # Skip if it looks like a constructor/destructor
            if func_name in ['if', 'while', 'for', 'switch']:
                continue
            
            # Find the function body by matching braces
            start_pos = match.end() - 1  # Start from opening brace
            brace_count = 1
            i = start_pos + 1
            
            while i < len(code) and brace_count > 0:
                # Handle string literals to avoid counting braces inside strings
                if code[i] == '"':
                    i += 1
                    while i < len(code) and code[i] != '"':
                        if code[i] == '\\':
                            i += 1  # Skip escape sequences
                        i += 1
                    if i < len(code):
                        i += 1
                    continue
                
                if code[i] == '{':
                    brace_count += 1
                elif code[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                body = code[start_pos+1:i-1]  # Extract body without braces
                
                # Extract return type (everything before function name)
                before_name = code[:match.start()].strip()
                return_type = before_name.split()[-1] if before_name else "void"
                
                functions.append({
                    'name': func_name,
                    'return_type': return_type,
                    'body': body,
                    'start': match.start(),
                    'end': i
                })
        
        return functions
    
    @staticmethod
    def _extract_cfg_from_function(func_info: Dict) -> CFG:
        """Extract CFG from a single function using simplified parsing"""
        function_name = func_info['name']
        body = func_info['body'].strip()
        
        cfg = CFG(function=function_name)
        node_id = 1
        
        # Create entry node
        entry_node = CFGNode(id=node_id, type="entry")
        cfg.nodes.append(entry_node)
        entry_id = node_id
        node_id += 1
        
        # Simple tokenization: split by common delimiters while preserving structure
        # Remove braces for now, process sequentially
        tokens = re.split(r'[;\{\}]', body)
        
        prev_node_id = entry_id
        condition_stack = []  # Track nested conditions
        
        for token in tokens:
            token = token.strip()
            if not token or len(token) < 2:
                continue
            
            # Detect if statements
            if_match = re.match(r'if\s*\(([^)]+)\)', token)
            if if_match:
                condition = if_match.group(1).strip()
                if condition:
                    cond_node = CFGNode(id=node_id, type="condition", expr=condition)
                    cfg.nodes.append(cond_node)
                    cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
                    
                    condition_stack.append({'node_id': node_id, 'type': 'if'})
                    prev_node_id = node_id
                    node_id += 1
                    continue
            
            # Detect else
            if re.match(r'else', token):
                if condition_stack:
                    cond_info = condition_stack[-1]
                    if cond_info['type'] == 'if':
                        # Add false branch edge
                        false_node_id = node_id
                        cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=false_node_id, label="false"))
                        prev_node_id = false_node_id
                        node_id += 1
                continue
            
            # Detect return statements
            if re.match(r'return', token):
                return_node = CFGNode(id=node_id, type="return")
                cfg.nodes.append(return_node)
                cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
                prev_node_id = node_id
                node_id += 1
                continue
            
            # Detect function calls (simplified - look for identifier followed by parentheses)
            call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'
            calls = re.findall(call_pattern, token)
            
            # Filter out C++ keywords
            cpp_keywords = {'if', 'while', 'for', 'switch', 'return', 'new', 'delete', 
                          'throw', 'try', 'catch', 'sizeof', 'typeid', 'static_cast',
                          'dynamic_cast', 'const_cast', 'reinterpret_cast'}
            
            for callee in calls:
                if callee not in cpp_keywords and len(callee) > 1:
                    call_node = CFGNode(id=node_id, type="call", callee=callee)
                    cfg.nodes.append(call_node)
                    cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
                    prev_node_id = node_id
                    node_id += 1
                    break  # Only take first call per token
            
            # For condition nodes, ensure true branch exists
            if condition_stack:
                top_cond = condition_stack[-1]
                if top_cond['type'] == 'if' and not any(e.from_node == top_cond['node_id'] and e.label == "true" for e in cfg.edges):
                    # Add true branch
                    true_node_id = node_id
                    cfg.edges.append(CFGEdge(from_node=top_cond['node_id'], to_node=true_node_id, label="true"))
                    prev_node_id = true_node_id
                    node_id += 1
                    # Pop condition after processing
                    condition_stack.pop()
        
        # Ensure all condition nodes have both branches
        for cond_info in condition_stack:
            if not any(e.from_node == cond_info['node_id'] and e.label == "true" for e in cfg.edges):
                true_node_id = node_id
                cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=true_node_id, label="true"))
                node_id += 1
            if not any(e.from_node == cond_info['node_id'] and e.label == "false" for e in cfg.edges):
                false_node_id = node_id
                cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=false_node_id, label="false"))
                node_id += 1
        
        # Ensure return node exists
        has_return = any(node.type == "return" for node in cfg.nodes)
        if not has_return:
            return_node = CFGNode(id=node_id, type="return")
            cfg.nodes.append(return_node)
            # Connect all nodes without outgoing edges to return
            nodes_with_outgoing = {e.from_node for e in cfg.edges}
            nodes_to_connect = [n.id for n in cfg.nodes if n.id not in nodes_with_outgoing and n.type != "return" and n.id != entry_id]
            if nodes_to_connect:
                for node_id_to_connect in nodes_to_connect:
                    cfg.edges.append(CFGEdge(from_node=node_id_to_connect, to_node=node_id))
            else:
                cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
        
        # Ensure entry node connects to something
        if not any(e.from_node == entry_id for e in cfg.edges):
            first_non_entry = [n.id for n in cfg.nodes if n.id != entry_id]
            if first_non_entry:
                cfg.edges.insert(0, CFGEdge(from_node=entry_id, to_node=first_non_entry[0]))
        
        return cfg
    
    @staticmethod
    def build_from_source(source_dir: str, function_name: Optional[str] = None) -> Optional[CFG]:
        """
        Build CFG from C++ source directory.
        
        Implements a basic C++ parser to extract CFG.
        Note: For production use, replace with Clang LibTooling integration.
        """
        # Find C++ files
        cpp_files = []
        for ext in ['*.cpp', '*.cc', '*.cxx', '*.c++', '*.hpp', '*.h']:
            cpp_files.extend(glob.glob(os.path.join(source_dir, ext)))
            cpp_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        
        if not cpp_files:
            print(f"  Warning: No C++ files found in {source_dir}")
            return None
        
        # Try to load pre-existing CFG JSON if available
        cfg_file = os.path.join(source_dir, 'cfg.json')
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, 'r') as f:
                    cfg_json = json.load(f)
                print(f"  [OK] Loaded existing CFG from: {cfg_file}")
                return parse_cfg(cfg_json)
            except Exception as e:
                print(f"  [WARNING] Could not load existing CFG: {e}")
        
        # Parse C++ files
        print(f"  [INFO] Parsing {min(len(cpp_files), 10)} C++ file(s) (limited for performance)...")
        print("  [NOTE] Using heuristic-based parser. For production, use Clang LibTooling.")
        
        all_functions = []
        for cpp_file in cpp_files[:10]:  # Limit to first 10 files for performance
            code = CFGBuilder._read_cpp_file(cpp_file)
            if not code:
                continue
            
            preprocessed = CFGBuilder._preprocess_code(code)
            functions = CFGBuilder._find_functions(preprocessed)
            
            for func in functions:
                # Only add functions with non-empty bodies
                if func['body'].strip():
                    func['file'] = os.path.basename(cpp_file)
                    all_functions.append(func)
        
        if not all_functions:
            print("  [WARNING] No functions with bodies found in C++ files")
            print("  [INFO] Using minimal placeholder CFG. For accurate results, use Clang LibTooling.")
            # Return a minimal CFG
            cfg = CFG(function=function_name or "unknown")
            cfg.nodes.append(CFGNode(id=1, type="entry"))
            cfg.nodes.append(CFGNode(id=2, type="return"))
            cfg.edges.append(CFGEdge(from_node=1, to_node=2))
            return cfg
        
        # Select function to analyze
        selected_func = None
        if function_name:
            for func in all_functions:
                if func['name'] == function_name or func['name'].endswith('::' + function_name):
                    selected_func = func
                    break
        
        if not selected_func:
            # Use first function found
            selected_func = all_functions[0]
            print(f"  [INFO] Analyzing function: {selected_func['name']} (from {selected_func['file']})")
            if len(all_functions) > 1:
                print(f"  [NOTE] Found {len(all_functions)} functions. Analyzing first one.")
                print(f"  [TIP] To analyze specific function, implement function name filtering.")
        
        # Extract CFG
        try:
            cfg = CFGBuilder._extract_cfg_from_function(selected_func)
            if cfg.nodes and cfg.edges:
                print(f"  [OK] Built CFG for function: {cfg.function}")
                print(f"  [NOTE] CFG extracted using heuristic parser. For production accuracy, use Clang LibTooling.")
            else:
                print(f"  [WARNING] CFG extraction produced empty result")
                # Return minimal CFG
                cfg = CFG(function=selected_func['name'])
                cfg.nodes.append(CFGNode(id=1, type="entry"))
                cfg.nodes.append(CFGNode(id=2, type="return"))
                cfg.edges.append(CFGEdge(from_node=1, to_node=2))
            return cfg
        except Exception as e:
            print(f"  [ERROR] Failed to extract CFG: {e}")
            print("  [INFO] Using minimal CFG. For accurate extraction, use Clang LibTooling integration.")
            import traceback
            if __debug__:
                traceback.print_exc()
            # Return minimal CFG
            cfg = CFG(function=selected_func['name'] if selected_func else "unknown")
            cfg.nodes.append(CFGNode(id=1, type="entry"))
            cfg.nodes.append(CFGNode(id=2, type="return"))
            cfg.edges.append(CFGEdge(from_node=1, to_node=2))
            return cfg
    
    @staticmethod
    def save_cfg(cfg: CFG, output_file: str = "cfg.json"):
        """Save CFG to JSON file"""
        cfg_json = {
            "function": cfg.function,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "expr": node.expr,
                    "callee": node.callee,
                    "label": node.label
                }
                for node in cfg.nodes
            ],
            "edges": [
                {
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "label": edge.label
                }
                for edge in cfg.edges
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(cfg_json, f, indent=2)


# ============================================================================
# MODULE 0.1: CALLGRAPH BUILDER
# ============================================================================

class CallGraphBuilder:
    """Builds CallGraph JSON from CFG or external source"""
    
    @staticmethod
    def build_from_cfg(cfg: CFG) -> CallGraph:
        """
        Build CallGraph from CFG call nodes.
        Calls must be derived ONLY from CFG call nodes.
        """
        cg = CallGraph()
        
        # Extract all call nodes from CFG
        calls = []
        for node in cfg.nodes:
            if node.type == "call" and node.callee:
                calls.append(node.callee)
        
        if cfg.function:
            cg.calls[cfg.function] = list(set(calls))  # Remove duplicates
        
        return cg
    
    @staticmethod
    def build_from_source(source_dir: str) -> Optional[CallGraph]:
        """
        Build CallGraph from C++ source directory.
        
        For now, tries to load pre-existing callgraph.json if available.
        TODO: Replace with Clang CallGraph extraction
        """
        callgraph_file = os.path.join(source_dir, 'callgraph.json')
        if os.path.exists(callgraph_file):
            try:
                with open(callgraph_file, 'r') as f:
                    cg_json = json.load(f)
                return parse_callgraph(cg_json)
            except Exception as e:
                print(f"Warning: Could not load existing callgraph.json: {e}")
        
        # Placeholder: Return empty CallGraph if cannot be built
        print("[WARNING] CallGraph Builder: Using placeholder. Replace with Clang integration.")
        return CallGraph()
    
    @staticmethod
    def save_callgraph(callgraph: CallGraph, output_file: str = "callgraph.json"):
        """Save CallGraph to JSON file"""
        cg_json = {
            "functions": callgraph.calls
        }
        
        with open(output_file, 'w') as f:
            json.dump(cg_json, f, indent=2)


# ============================================================================
# MODULE 0.2: DESCRIPTION BUILDER
# ============================================================================

class DescriptionBuilder:
    """Builds initial Description JSON automatically"""
    
    @staticmethod
    def build_from_cfg(cfg: CFG, callgraph: CallGraph) -> Description:
        """
        Create initial Description JSON from CFG and CallGraph.
        
        Rules:
        - Neutral and minimal
        - Non-speculative
        - No control-flow claims
        """
        desc = Description(function=cfg.function)
        
        # Build minimal summary from function name
        desc.summary = f"Function {cfg.function}"
        
        # Build notes from available information (minimal, non-speculative)
        notes_parts = []
        
        # Add call information if available
        if cfg.function in callgraph.calls and callgraph.calls[cfg.function]:
            calls_str = ", ".join(callgraph.calls[cfg.function])
            notes_parts.append(f"Calls: {calls_str}")
        
        # Add condition information if available (just list, don't interpret)
        conditions = [node.expr for node in cfg.nodes if node.type == "condition" and node.expr]
        if conditions:
            conditions_str = ", ".join(conditions)
            notes_parts.append(f"Conditions present: {conditions_str}")
        
        desc.notes = ". ".join(notes_parts) if notes_parts else "Not present in CFG"
        desc.validated = False  # Not yet validated
        
        return desc
    
    @staticmethod
    def save_description(desc: Description, output_file: str = "description.json"):
        """Save Description to JSON file"""
        desc_json = {
            desc.function: {
                "summary": desc.summary,
                "notes": desc.notes,
                "validated": desc.validated,
                "issues": desc.issues
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(desc_json, f, indent=2)


# ============================================================================
# MODULE 4: CFG VALIDATION AGENT
# ============================================================================

class CFGValidationAgent:
    """Validates CFG structure and correctness"""
    
    @staticmethod
    def validate(cfg: CFG) -> Tuple[bool, List[str]]:
        """
        Validate CFG structure.
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Check if CFG has function name
        if not cfg.function:
            issues.append("CFG missing function name")
        
        # Check if CFG has nodes
        if not cfg.nodes:
            issues.append("CFG has no nodes")
            return False, issues
        
        # Check for entry node
        entry_nodes = [n for n in cfg.nodes if n.type == "entry"]
        if not entry_nodes:
            issues.append("CFG missing entry node")
        elif len(entry_nodes) > 1:
            issues.append(f"CFG has multiple entry nodes: {len(entry_nodes)}")
        
        # Check node IDs are unique
        node_ids = [node.id for node in cfg.nodes]
        if len(node_ids) != len(set(node_ids)):
            issues.append("CFG has duplicate node IDs")
        
        # Check edge references
        valid_node_ids = set(node_ids)
        for edge in cfg.edges:
            if edge.from_node not in valid_node_ids:
                issues.append(f"Edge references invalid from_node: {edge.from_node}")
            if edge.to_node not in valid_node_ids:
                issues.append(f"Edge references invalid to_node: {edge.to_node}")
        
        # Check condition nodes have expressions
        for node in cfg.nodes:
            if node.type == "condition" and not node.expr:
                issues.append(f"Condition node {node.id} missing expression")
        
        # Check call nodes have callees
        for node in cfg.nodes:
            if node.type == "call" and not node.callee:
                issues.append(f"Call node {node.id} missing callee")
        
        # Check for unreachable nodes (nodes with no incoming edges except entry)
        entry_ids = {n.id for n in entry_nodes}
        reachable_nodes = set(entry_ids)
        for edge in cfg.edges:
            reachable_nodes.add(edge.to_node)
        
        unreachable = valid_node_ids - reachable_nodes
        if unreachable:
            issues.append(f"CFG has unreachable nodes: {unreachable}")
        
        return len(issues) == 0, issues


# ============================================================================
# MODULE 5: CALLGRAPH VALIDATION AGENT
# ============================================================================

class CallGraphValidationAgent:
    """Validates CallGraph structure"""
    
    @staticmethod
    def validate(callgraph: CallGraph, cfg: CFG) -> Tuple[bool, List[str]]:
        """
        Validate CallGraph against CFG.
        
        Returns:
            (is_valid, issues)
        """
        issues = []
        
        # Extract calls from CFG
        cfg_calls = set()
        for node in cfg.nodes:
            if node.type == "call" and node.callee:
                cfg_calls.add(node.callee)
        
        # Check if CallGraph has the function
        if cfg.function not in callgraph.calls:
            # This is OK if function has no calls
            if cfg_calls:
                issues.append(f"CallGraph missing function {cfg.function} but CFG has calls")
        else:
            # Check if CallGraph calls match CFG calls
            cg_calls = set(callgraph.calls[cfg.function])
            extra_calls = cg_calls - cfg_calls
            if extra_calls:
                issues.append(f"CallGraph has calls not in CFG: {extra_calls}")
        
        return len(issues) == 0, issues


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

def build_from_source(source_dir: str, reuse_json: bool = False, dry_run: bool = False):
    """Build CFG, CallGraph, and Description from C++ source directory"""
    print(f"[INFO] Building from source directory: {source_dir}\n")
    
    # Step 1: Load C++ source directory
    if not os.path.isdir(source_dir):
        if source_dir.startswith('http://') or source_dir.startswith('https://'):
            print(f"Error: Agent requires a local directory path, not a URL.")
            print(f"  Please clone the repository first:")
            print(f"    git clone {source_dir}")
            print(f"  Then run:")
            print(f"    python agent.py <local_cloned_directory>")
        else:
            print(f"Error: Source directory does not exist: {source_dir}")
            print(f"  Make sure the path is correct and points to a local directory.")
        sys.exit(1)
    
    # Step 2: Build CFG JSON
    print("Step 2: Building CFG JSON...")
    cfg = None
    
    if reuse_json:
        cfg_file = os.path.join(source_dir, "cfg.json")
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, 'r') as f:
                    cfg_json = json.load(f)
                cfg = parse_cfg(cfg_json)
                print(f"  [OK] Loaded existing CFG from: {cfg_file}")
            except Exception as e:
                print(f"  [WARNING] Could not load existing CFG: {e}")
    
    if cfg is None:
        cfg = CFGBuilder.build_from_source(source_dir)
        if cfg is None:
            print("  [ERROR] Cannot build CFG. Aborting.")
            sys.exit(1)
        print(f"  [OK] Built CFG for function: {cfg.function}")
    
    print(f"  Nodes: {len(cfg.nodes)}, Edges: {len(cfg.edges)}")
    
    # Step 3: Validate CFG JSON
    print("\nStep 3: Validating CFG JSON...")
    cfg_validator = CFGValidationAgent()
    is_valid, issues = cfg_validator.validate(cfg)
    
    if not is_valid:
        print("  [ERROR] CFG validation FAILED:")
        for issue in issues:
            print(f"    - {issue}")
        print("  Aborting due to CFG validation failure.")
        sys.exit(1)
    
    print("  [OK] CFG validation PASSED")
    
    # Save CFG if not dry run
    if not dry_run:
        cfg_output = os.path.join(source_dir, "cfg.json")
        CFGBuilder.save_cfg(cfg, cfg_output)
        print(f"  [OK] Saved CFG to: {cfg_output}")
    
    # Step 4: Build CallGraph JSON
    print("\nStep 4: Building CallGraph JSON...")
    callgraph = None
    
    if reuse_json:
        cg_file = os.path.join(source_dir, "callgraph.json")
        if os.path.exists(cg_file):
            try:
                with open(cg_file, 'r') as f:
                    cg_json = json.load(f)
                callgraph = parse_callgraph(cg_json)
                print(f"  [OK] Loaded existing CallGraph from: {cg_file}")
            except Exception as e:
                print(f"  [WARNING] Could not load existing CallGraph: {e}")
    
    if callgraph is None:
        # First try to build from source, then fall back to CFG
        callgraph = CallGraphBuilder.build_from_source(source_dir)
        if not callgraph.calls:
            callgraph = CallGraphBuilder.build_from_cfg(cfg)
            print("  [OK] Built CallGraph from CFG")
        else:
            print("  [OK] Built CallGraph from source")
    
    print(f"  Functions: {len(callgraph.calls)}")
    
    # Step 5: Validate CallGraph JSON
    print("\nStep 5: Validating CallGraph JSON...")
    cg_validator = CallGraphValidationAgent()
    is_valid, issues = cg_validator.validate(callgraph, cfg)
    
    if not is_valid:
        print("  [WARNING] CallGraph validation found issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] CallGraph validation PASSED")
    
    # Save CallGraph if not dry run
    if not dry_run:
        cg_output = os.path.join(source_dir, "callgraph.json")
        CallGraphBuilder.save_callgraph(callgraph, cg_output)
        print(f"  [OK] Saved CallGraph to: {cg_output}")
    
    # Step 6: Build initial Description JSON
    print("\nStep 6: Building initial Description JSON...")
    desc = DescriptionBuilder.build_from_cfg(cfg, callgraph)
    print(f"  [OK] Built Description for: {desc.function}")
    
    # Step 7: Validate & correct Description JSON
    print("\nStep 7: Validating & correcting Description JSON...")
    desc_validator = DescriptionValidationAgent()
    is_valid, corrected_desc, justification = desc_validator.validate(desc, cfg, callgraph)
    
    if not is_valid:
        print(f"  [WARNING] Description validation FAILED: {justification}")
        desc = corrected_desc
        print("  [OK] Description corrected")
    else:
        print(f"  [OK] Description validation PASSED: {justification}")
    
    # Save Description if not dry run
    if not dry_run:
        desc_output = os.path.join(source_dir, "description.json")
        DescriptionBuilder.save_description(desc, desc_output)
        print(f"  [OK] Saved Description to: {desc_output}")
    
    # Step 8: Generate Mermaid diagram
    print("\nStep 8: Generating Mermaid diagram...")
    primary_agent = PrimaryAnalysisAgent()
    mermaid = primary_agent.generate_mermaid(cfg, desc)
    print("  [OK] Generated Mermaid diagram")
    
    # Step 9: Validate Mermaid diagram
    print("\nStep 9: Validating Mermaid diagram...")
    diagram_validator = DiagramValidationAgent()
    is_valid, corrected_mermaid, issues = diagram_validator.validate(mermaid, cfg)
    
    validation_attempts = 1
    while not is_valid and validation_attempts < 3:
        print(f"  [WARNING] Diagram validation FAILED (attempt {validation_attempts})")
        for issue in issues:
            print(f"    - {issue}")
        
        mermaid = corrected_mermaid
        is_valid, corrected_mermaid, issues = diagram_validator.validate(mermaid, cfg)
        validation_attempts += 1
    
    if is_valid:
        print("  [OK] Diagram validation PASSED")
    else:
        print(f"  [ERROR] Diagram validation FAILED after {validation_attempts} attempts")
        print("  Aborting due to persistent validation failures.")
        sys.exit(1)
    
    # Step 10: Output
    print("\n" + "=" * 50)
    print("=== FINAL MERMAID DIAGRAM ===")
    print("=" * 50 + "\n")
    print(mermaid)
    
    # Save outputs if not dry run
    if not dry_run:
        output_files = {
            "cfg.json": os.path.join(source_dir, "cfg.json"),
            "callgraph.json": os.path.join(source_dir, "callgraph.json"),
            "description.json": os.path.join(source_dir, "description.json"),
            "output.mermaid": os.path.join(source_dir, "output.mermaid")
        }
        
        # Save mermaid
        with open(output_files["output.mermaid"], 'w') as f:
            f.write(mermaid + "\n")
        
        print("\n=== GENERATED FILES ===")
        for name, path in output_files.items():
            print(f"  {name}: {path}")
    
    # Short explanation
    print("\n=== EXPLANATION ===")
    print(f"Function: {cfg.function}")
    print(f"Summary: {desc.summary}")
    print(f"Flow: {len(cfg.nodes)} nodes, {len(cfg.edges)} edges")
    if desc.notes:
        print(f"Notes: {desc.notes}")


def load_from_json(cfg_file: str, desc_file: str, callgraph_file: Optional[str] = None):
    """Load from JSON files (backward compatible mode)"""
    print("[INFO] Loading from JSON files (backward compatible mode)\n")
    
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


def main():
    """Main execution flow - supports both building from source and loading from JSON"""
    parser = argparse.ArgumentParser(
        description="LLM-Assisted C++ Program Understanding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from C++ source directory
  python agent.py /path/to/cpp/source
  
  # Load from JSON files (backward compatible)
  python agent.py example_cfg.json example_description.json example_callgraph.json
  
  # Build with options
  python agent.py /path/to/cpp/source --reuse-json --dry-run
        """
    )
    
    parser.add_argument('input', nargs='+', help='Source directory OR JSON files (cfg, description, [callgraph])')
    parser.add_argument('--reuse-json', action='store_true', help='Reuse existing JSON files if available')
    parser.add_argument('--dry-run', action='store_true', help='Do not save output files')
    
    args = parser.parse_args()
    
    # Determine mode: if single argument and it's a directory, use build mode
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        # Build from source mode
        build_from_source(args.input[0], args.reuse_json, args.dry_run)
    elif len(args.input) >= 2:
        # Load from JSON mode (backward compatible)
        cfg_file = args.input[0]
        desc_file = args.input[1]
        callgraph_file = args.input[2] if len(args.input) > 2 else None
        load_from_json(cfg_file, desc_file, callgraph_file)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

