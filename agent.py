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

# Clang integration
try:
    from clang.cindex import Index, TranslationUnit, Cursor, CursorKind, SourceLocation, Config
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    Config = None
    print("[ERROR] clang.cindex not available. Install with: pip install clang")


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


@dataclass
class FunctionInfo:
    """Information about a discovered function"""
    name: str
    source_file: str
    full_name: str = ""  # with namespace/class
    cfg: Optional[CFG] = None
    description: Optional[Description] = None


@dataclass
class FunctionRegistry:
    """Registry to track all discovered functions"""
    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    
    def add_function(self, func_info: FunctionInfo):
        """Add a function to the registry"""
        self.functions[func_info.name] = func_info
    
    def get_function(self, name: str) -> Optional[FunctionInfo]:
        """Get function by name"""
        return self.functions.get(name)
    
    def get_all_functions(self) -> List[FunctionInfo]:
        """Get all registered functions"""
        return list(self.functions.values())


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
# MODULE A: CLANG INTEGRATION LAYER
# ============================================================================

class ClangIntegration:
    """Clang integration for AST traversal and CFG extraction"""
    
    # Clang library paths to try (in order)
    CLANG_PATHS = [
        "/usr/lib/llvm-18/lib/libclang.so",  # User's specified path (highest priority)
        "/usr/lib/llvm-18/lib/libclang.so.18",
        "/usr/lib/llvm-17/lib/libclang.so",
        "/usr/lib/llvm-17/lib/libclang.so.17",
        "/usr/lib/llvm-16/lib/libclang.so",
        "/usr/lib/llvm-16/lib/libclang.so.16",
        "/usr/lib/llvm-15/lib/libclang.so",
        "/usr/lib/llvm-15/lib/libclang.so.15",
        "/usr/lib/x86_64-linux-gnu/libclang.so.1",
        "/usr/local/lib/libclang.so",
    ]
    
    _clang_initialized = False
    
    @staticmethod
    def _initialize_clang() -> bool:
        """Initialize Clang with library path detection"""
        if ClangIntegration._clang_initialized:
            return True
        
        if not CLANG_AVAILABLE:
            return False
        
        # Try to set Clang library path
        clang_found = False
        for path in ClangIntegration.CLANG_PATHS:
            if os.path.exists(path):
                try:
                    if Config:
                        Config.set_library_file(path)
                    print(f"[INFO] Using Clang library: {path}")
                    clang_found = True
                    break
                except Exception as e:
                    # Try next path
                    continue
        
        if not clang_found:
            # Try environment variable
            env_path = os.environ.get('LIBCLANG_LIBRARY_PATH')
            if env_path and os.path.exists(env_path):
                try:
                    if Config:
                        Config.set_library_file(env_path)
                    print(f"[INFO] Using Clang library from environment: {env_path}")
                    clang_found = True
                except Exception:
                    pass
        
        # Test if Clang works
        try:
            Index.create()
            ClangIntegration._clang_initialized = True
            return True
        except Exception as e:
            if not clang_found:
                print(f"[ERROR] Clang initialization failed: {e}")
                print(f"[ERROR] Tried paths:")
                for path in ClangIntegration.CLANG_PATHS[:5]:
                    exists = "✓" if os.path.exists(path) else "✗"
                    print(f"  {exists} {path}")
                print(f"\n[ERROR] Please ensure Clang is installed and library is accessible.")
                print(f"  Or set LIBCLANG_LIBRARY_PATH environment variable.")
            return False
    
    @staticmethod
    def check_clang_available() -> bool:
        """Check if Clang is available"""
        return ClangIntegration._initialize_clang()
    
    @staticmethod
    def _is_user_defined_function(cursor: Cursor, project_root: str) -> bool:
        """
        Strict filtering: Only accept user-defined functions from project path.
        
        Returns True only if:
        - Function is a definition (not just declaration)
        - Function location file is within project_root
        - Function name doesn't start with "__" (compiler builtins)
        - Function name doesn't start with "operator" (operator overloads)
        - Function name doesn't contain "<" or ">" (template instantiations)
        - Function is not implicit
        - Function is not from system header
        """
        # Must be a function declaration or method
        if cursor.kind not in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD]:
            return False
        
        # Must be a definition, not just declaration (check if method exists)
        if hasattr(cursor, 'is_definition'):
            if not cursor.is_definition():
                return False
        else:
            # If is_definition() doesn't exist, skip (likely just a declaration)
            # We can't tell if it's a definition, so err on the side of caution
            return False
        
        # Must not be implicit (compiler-generated) - check if method exists
        if hasattr(cursor, 'is_implicit') and cursor.is_implicit():
            return False
        
        # Get function name
        func_name = cursor.spelling
        if not func_name or len(func_name) == 0:
            return False
        
        # Reject compiler builtins (names starting with __)
        if func_name.startswith("__"):
            return False
        
        # Reject operator overloads
        if func_name.startswith("operator"):
            return False
        
        # Reject template instantiations (contain < or >)
        if "<" in func_name or ">" in func_name:
            return False
        
        # Reject anonymous functions
        if func_name.startswith("_Z") or func_name.startswith("__Z"):  # Name mangling
            return False
        
        # Get file location - must be within project root
        location = cursor.location
        if not location:
            return False
        
        file_obj = location.file
        if not file_obj:
            return False
        
        file_path = file_obj.name
        if not file_path:
            return False
        
        # Normalize paths for comparison (handle Windows paths)
        file_path_abs = os.path.normpath(os.path.abspath(file_path))
        project_root_abs = os.path.normpath(os.path.abspath(project_root))
        
        # Convert to lowercase for comparison on Windows (case-insensitive filesystem)
        if os.name == 'nt':  # Windows
            file_path_abs = file_path_abs.lower()
            project_root_abs = project_root_abs.lower()
        
        # File must be inside project root
        if not file_path_abs.startswith(project_root_abs):
            return False
        
        # Check if it's a system header (additional safety check)
        # System headers typically have paths like /usr/include, /usr/lib, etc.
        # On Windows, check for system paths too
        system_paths = ['/usr/include', '/usr/lib', '/usr/local/include', 
                       '/usr/local/lib', '/opt', '/usr/share',
                       '\\windows\\', '\\program files\\', 'c:\\windows']
        file_path_lower = file_path_abs.lower()
        for sys_path in system_paths:
            if sys_path.lower() in file_path_lower:
                return False
        
        return True
    
    @staticmethod
    def discover_all_functions(source_dir: str, compile_args_map: Dict[str, List[str]] = None) -> List[FunctionInfo]:
        """
        Discover all user-defined function definitions using Clang AST.
        
        STRICT FILTERING: Only functions from project path, no system/STL functions.
        
        Returns list of FunctionInfo objects.
        """
        if not ClangIntegration.check_clang_available():
            raise RuntimeError(
                "Clang is not available. Please ensure:\n"
                "  1. libclang is installed (e.g., libclang-dev or llvm packages)\n"
                "  2. Python clang package is installed: pip install clang\n"
                "  3. Clang library exists at one of the expected paths\n"
                f"     (Checked: {', '.join(ClangIntegration.CLANG_PATHS[:3])})\n"
                "  4. Or set LIBCLANG_LIBRARY_PATH environment variable\n"
                "DO NOT continue without Clang - this is required."
            )
        
        # Normalize project root path (handle Windows paths)
        project_root = os.path.normpath(os.path.abspath(source_dir))
        
        functions = []
        index = Index.create()
        
        # Find all C++ files ONLY within project directory
        cpp_files = []
        for ext in ['*.cpp', '*.cc', '*.cxx', '*.c++']:
            # Only search within project directory
            pattern = os.path.join(project_root, ext)
            cpp_files.extend(glob.glob(pattern))
            # Recursive search but only within project
            pattern_recursive = os.path.join(project_root, '**', ext)
            cpp_files.extend(glob.glob(pattern_recursive, recursive=True))
        
        # Filter to only files actually in project root (normalize for comparison)
        normalized_project = project_root.lower() if os.name == 'nt' else project_root
        filtered_files = []
        for f in cpp_files:
            f_abs = os.path.normpath(os.path.abspath(f))
            f_normalized = f_abs.lower() if os.name == 'nt' else f_abs
            if f_normalized.startswith(normalized_project):
                filtered_files.append(f)
        cpp_files = filtered_files
        
        if not cpp_files:
            print(f"[WARNING] No C++ files found in {source_dir}")
            return functions
        
        # REQUIRE compile_commands.json
        compile_commands_path = os.path.join(source_dir, 'compile_commands.json')
        if not os.path.exists(compile_commands_path):
            # Check parent directories (up to 3 levels)
            checked_paths = [compile_commands_path]
            current_dir = source_dir
            for _ in range(3):
                parent = os.path.dirname(current_dir)
                if parent == current_dir:
                    break
                compile_commands_path = os.path.join(parent, 'compile_commands.json')
                checked_paths.append(compile_commands_path)
                if os.path.exists(compile_commands_path):
                    break
                current_dir = parent
        
        if not os.path.exists(compile_commands_path):
            error_msg = (
                f"[ERROR] compile_commands.json is REQUIRED but not found.\n"
                f"  Searched in:\n"
            )
            for path in checked_paths[:5]:
                error_msg += f"    - {path}\n"
            error_msg += (
                f"\n  Please generate compile_commands.json:\n"
                f"    - cmake: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .\n"
                f"    - bear: bear make\n"
                f"    - clang-tidy: clang-tidy -p .\n"
            )
            raise RuntimeError(error_msg)
        
        # Load compile_commands.json
        compile_commands = None
        try:
            with open(compile_commands_path, 'r') as f:
                compile_commands = json.load(f)
            print(f"  [OK] Found and loaded compile_commands.json: {compile_commands_path}")
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to parse compile_commands.json: {e}\n"
                f"  Path: {compile_commands_path}\n"
                f"  Please fix or regenerate compile_commands.json"
            )
        
        if not compile_commands or not isinstance(compile_commands, list):
            raise RuntimeError(
                f"[ERROR] compile_commands.json is invalid or empty\n"
                f"  Expected: array of compile command objects"
            )
        
        # Build file -> compile args mapping
        compile_args_map = {}
        for entry in compile_commands:
            file_path = entry.get('file', '')
            if file_path:
                # Normalize path
                abs_file = os.path.abspath(file_path)
                compile_args_map[abs_file] = entry.get('arguments', entry.get('command', '').split())
                # Also map by basename
                basename = os.path.basename(file_path)
                compile_args_map[basename] = entry.get('arguments', entry.get('command', '').split())
        
        print(f"  [INFO] Parsing {len(cpp_files)} C++ file(s) with Clang...")
        print(f"  [INFO] Project root: {project_root}")
        
        rejected_count = 0
        system_functions = []
        parse_errors = []
        
        for cpp_file in cpp_files[:50]:  # Limit to 50 files for performance
            try:
                # Get compile arguments for this file
                cpp_file_abs = os.path.abspath(cpp_file)
                file_args = None
                if compile_args_map:
                    file_args = compile_args_map.get(cpp_file_abs) or compile_args_map.get(os.path.basename(cpp_file))
                
                if file_args:
                    # Use compile arguments from compile_commands.json
                    # Remove compiler name (first arg) and file path (last arg typically)
                    parse_args = [arg for arg in file_args if arg not in ['-o', '-c'] and not arg.endswith('.o')]
                    # Ensure -x c++ is present
                    if '-x' not in parse_args:
                        parse_args.extend(['-x', 'c++'])
                else:
                    # Fallback: basic args (should not happen with proper compile_commands.json)
                    print(f"  [WARNING] No compile args found for {os.path.basename(cpp_file)}, using defaults")
                    parse_args = ['-std=c++17', '-x', 'c++']
                
                # Parse translation unit with proper options
                parse_options = 0
                try:
                    parse_options |= TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                except AttributeError:
                    pass
                
                try:
                    parse_options |= TranslationUnit.PARSE_INCOMPLETE
                except AttributeError:
                    pass
                
                # Parse with compile arguments from compile_commands.json
                tu = index.parse(cpp_file, args=parse_args, options=parse_options)
                
                # Check for Clang diagnostics (errors/warnings)
                diags = list(tu.diagnostics)
                if diags:
                    errors = [d for d in diags if d.severity >= 2]  # Error or Fatal
                    if errors:
                        error_msgs = []
                        for diag in errors[:5]:  # Limit to first 5 errors
                            error_msgs.append(f"    {diag.spelling} (line {diag.location.line})")
                        if len(errors) > 5:
                            error_msgs.append(f"    ... and {len(errors) - 5} more errors")
                        print(f"  [ERROR] Clang parsing errors in {os.path.basename(cpp_file)}:")
                        for msg in error_msgs:
                            print(msg)
                        # Continue but note that parsing may be incomplete
                
                # Walk AST to find function definitions
                def visit_node(cursor: Cursor):
                    nonlocal rejected_count
                    
                    try:
                        # Skip system headers early (normalize paths for comparison)
                        if cursor.location and cursor.location.file:
                            file_path = cursor.location.file.name
                            if file_path:
                                file_path_abs = os.path.normpath(os.path.abspath(file_path))
                                project_root_normalized = project_root.lower() if os.name == 'nt' else project_root
                                file_path_normalized = file_path_abs.lower() if os.name == 'nt' else file_path_abs
                                if not file_path_normalized.startswith(project_root_normalized):
                                    # Skip entire subtrees from system headers
                                    return
                        
                        # Check if it's a user-defined function (support multiple kinds)
                        if cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD,
                                          CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR]:
                            if ClangIntegration._is_user_defined_function(cursor, project_root):
                                func_name = cursor.spelling
                                
                                # Get file location for verification
                                location = cursor.location
                                file_path = location.file.name if location and location.file else ""
                                
                                # Get full qualified name
                                full_name = ClangIntegration._get_qualified_name(cursor)
                                
                                func_info = FunctionInfo(
                                    name=func_name,
                                    source_file=os.path.basename(file_path) if file_path else os.path.basename(cpp_file),
                                    full_name=full_name
                                )
                                functions.append(func_info)
                            else:
                                # Track rejected functions for validation
                                func_name = cursor.spelling
                                if func_name:
                                    try:
                                        location = cursor.location
                                        if location and location.file:
                                            file_path = location.file.name
                                            if file_path:
                                                file_path_abs = os.path.normpath(os.path.abspath(file_path))
                                                project_root_normalized = project_root.lower() if os.name == 'nt' else project_root
                                                file_path_normalized = file_path_abs.lower() if os.name == 'nt' else file_path_abs
                                                if not file_path_normalized.startswith(project_root_normalized):
                                                    system_functions.append(func_name)
                                                    rejected_count += 1
                                    except (AttributeError, TypeError):
                                        pass
                        
                        # Recurse into children (always attempt even if parent had issues)
                        try:
                            for child in cursor.get_children():
                                visit_node(child)
                        except (AttributeError, TypeError):
                            pass
                    except (AttributeError, TypeError) as attr_err:
                        # Skip nodes that cause attribute errors (version compatibility)
                        # But try to continue processing children
                        try:
                            for child in cursor.get_children():
                                visit_node(child)
                        except:
                            pass
                    except Exception as node_err:
                        # Log but don't stop processing - try to continue with children
                        try:
                            for child in cursor.get_children():
                                visit_node(child)
                        except:
                            pass
                
                # Call visit_node with outer error handling
                try:
                    visit_node(tu.cursor)
                except Exception as visit_err:
                    # If visit_node itself fails completely, log but continue
                    error_msg = str(visit_err)
                    if 'is_implicit' not in error_msg and 'object has no attribute' not in error_msg:
                        parse_errors.append((cpp_file, str(visit_err)))
                    continue
                
            except Exception as e:
                error_msg = str(e)
                # Store error for later reporting
                parse_errors.append((cpp_file, error_msg))
                # Only warn about critical errors (attribute errors are handled above)
                if 'is_implicit' in error_msg or 'object has no attribute' in error_msg:
                    # These are version compatibility issues - silently continue
                    pass
                else:
                    print(f"  [WARNING] Failed to parse {cpp_file}: {e}")
                # Continue processing other files
                continue
        
        # Validation: Check for too many system functions
        total_attempted = len(functions) + rejected_count
        if total_attempted > 0:
            system_ratio = rejected_count / total_attempted
            if system_ratio > 0.1:  # More than 10% are system functions
                print(f"  [WARNING] High ratio of system functions detected: {system_ratio:.1%}")
                print(f"    User functions: {len(functions)}, System/rejected: {rejected_count}")
        
        # Check for operator overloads (should be rejected by filter, but verify)
        operator_overloads = [f for f in functions if f.name.startswith("operator")]
        if operator_overloads:
            print(f"  [ERROR] Operator overloads detected (should be filtered): {[f.name for f in operator_overloads[:5]]}")
            # Remove them
            functions = [f for f in functions if not f.name.startswith("operator")]
        
        # Report parse errors if any
        if parse_errors:
            unique_errors = {}
            for file, error in parse_errors:
                if error not in unique_errors:
                    unique_errors[error] = []
                unique_errors[error].append(os.path.basename(file))
            
            for error, files in unique_errors.items():
                if 'is_implicit' in error or 'object has no attribute' in error:
                    print(f"  [INFO] Skipped {len(files)} file(s) due to Clang version compatibility")
                    if len(files) <= 3:
                        print(f"    Files: {', '.join(files)}")
                else:
                    print(f"  [WARNING] Parse errors in {len(files)} file(s): {error[:100]}")
        
        # Final validation
        if len(functions) == 0:
            print(f"  [ERROR] No user-defined functions found in project path")
            print(f"  [ERROR] Project root: {project_root}")
            print(f"  [ERROR] Files searched: {len(cpp_files)}")
            if cpp_files:
                print(f"  [ERROR] Example files: {[os.path.basename(f) for f in cpp_files[:5]]}")
            print(f"  [ERROR] This may indicate:")
            print(f"    - Filtering too strict")
            print(f"    - Path mismatch (check if files are actually in project root)")
            print(f"    - Clang parsing failures")
            raise RuntimeError("No valid user-defined functions found")
        
        if len(functions) == 1:
            print(f"  [WARNING] Only one function detected - this may indicate a problem")
        
        if len(functions) > 0:
            print(f"  [OK] Discovered {len(functions)} user-defined function(s)")
            if rejected_count > 0:
                print(f"  [INFO] Filtered out {rejected_count} system/STL/compiler functions")
            
            # Show some example functions found
            example_funcs = [f.name for f in functions[:5]]
            print(f"  [INFO] Example functions: {', '.join(example_funcs)}")
        else:
            print(f"  [WARNING] No functions discovered yet")
            if cpp_files:
                print(f"  [INFO] Searched {len(cpp_files)} file(s), but no valid functions found")
                print(f"  [INFO] This may indicate:")
                print(f"    - All functions are filtered out (check filtering logic)")
                print(f"    - Clang parsing issues (check errors above)")
                print(f"    - Files don't contain function definitions")
        
        return functions
    
    @staticmethod
    def _get_qualified_name(cursor: Cursor) -> str:
        """Get fully qualified function name"""
        parts = []
        current = cursor
        
        while current:
            if current.kind == CursorKind.FUNCTION_DECL:
                parts.insert(0, current.spelling)
            elif current.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                parts.insert(0, current.spelling)
            elif current.kind == CursorKind.NAMESPACE:
                parts.insert(0, current.spelling)
            current = current.semantic_parent
        
        return '::'.join(filter(None, parts))
    
    @staticmethod
    def build_cfg_from_clang(source_file: str, function_name: str, compile_args_map: Dict[str, List[str]] = None) -> Optional[CFG]:
        """
        Build CFG for a specific function using Clang's AST.
        
        REQUIRES compile_commands.json for proper parsing.
        NO FALLBACK - fails clearly if CFG cannot be extracted.
        """
        if not ClangIntegration.check_clang_available():
            raise RuntimeError("Clang is not available - cannot build CFG")
        
        index = Index.create()
        
        # Get compile arguments from compile_commands.json
        source_file_abs = os.path.abspath(source_file)
        args = None
        
        if compile_args_map:
            args = compile_args_map.get(source_file_abs) or compile_args_map.get(os.path.basename(source_file))
        
        if not args:
            # Try to find compile_commands.json
            source_dir = os.path.dirname(source_file_abs)
            compile_commands_path = os.path.join(source_dir, 'compile_commands.json')
            if not os.path.exists(compile_commands_path):
                # Check parent directories
                for _ in range(3):
                    parent = os.path.dirname(source_dir)
                    if parent == source_dir:
                        break
                    compile_commands_path = os.path.join(parent, 'compile_commands.json')
                    if os.path.exists(compile_commands_path):
                        break
                    source_dir = parent
            
            if os.path.exists(compile_commands_path):
                try:
                    with open(compile_commands_path, 'r') as f:
                        compile_commands = json.load(f)
                    for entry in compile_commands:
                        if entry.get('file') == source_file_abs or os.path.basename(entry.get('file', '')) == os.path.basename(source_file):
                            args = entry.get('arguments', entry.get('command', '').split())
                            break
                except Exception:
                    pass
        
        if not args:
            raise RuntimeError(
                f"[ERROR] Cannot find compile arguments for {source_file}\n"
                f"  compile_commands.json is REQUIRED for CFG extraction"
            )
        
        # Remove output flags and ensure proper parsing
        parse_args = [arg for arg in args if arg not in ['-o', '-c'] and not arg.endswith('.o')]
        if '-x' not in parse_args:
            parse_args.extend(['-x', 'c++'])
        
        # Parse with proper options
        parse_options = 0
        try:
            parse_options |= TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        except AttributeError:
            pass
        
        try:
            tu = index.parse(source_file, args=parse_args, options=parse_options)
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] Failed to parse translation unit for {source_file}: {e}\n"
                f"  This indicates Clang cannot parse the file correctly"
            )
        
        # Check diagnostics
        diags = list(tu.diagnostics)
        errors = [d for d in diags if d.severity >= 2]  # Error or Fatal
        if errors:
            error_msgs = [f"    {d.spelling} (line {d.location.line})" for d in errors[:5]]
            raise RuntimeError(
                f"[ERROR] Clang parsing errors in {source_file}:\n" +
                "\n".join(error_msgs) +
                f"\n  Cannot extract CFG with parsing errors"
            )
        
        target_func = None
        
        # Find the function (support multiple cursor kinds)
        def find_function(cursor: Cursor):
            nonlocal target_func
            if target_func:
                return
            
            if cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD, 
                              CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR]:
                if hasattr(cursor, 'is_definition') and cursor.is_definition():
                    if cursor.spelling == function_name:
                        target_func = cursor
                        return
            
            for child in cursor.get_children():
                find_function(child)
        
        find_function(tu.cursor)
        
        if not target_func:
            raise RuntimeError(
                f"[ERROR] Function '{function_name}' not found in {source_file}\n"
                f"  Function may not exist or may be filtered out"
            )
        
        # Extract CFG using AST traversal
        try:
            cfg = ClangIntegration._extract_cfg_from_cursor(target_func, function_name)
            if not cfg or len(cfg.nodes) < 2:
                raise RuntimeError(
                    f"[ERROR] CFG extraction produced invalid result for {function_name}\n"
                    f"  Nodes: {len(cfg.nodes) if cfg else 0}, Expected: >= 2"
                )
            return cfg
        except Exception as e:
            raise RuntimeError(
                f"[ERROR] CFG extraction failed for {function_name}: {e}\n"
                f"  Cannot generate CFG from AST"
            )
    
    @staticmethod
    def _extract_cfg_from_cursor(cursor: Cursor, function_name: str) -> CFG:
        """Extract CFG from Clang cursor using AST traversal"""
        cfg = CFG(function=function_name)
        node_id = 1
        
        # Create entry node
        entry_node = CFGNode(id=node_id, type="entry")
        cfg.nodes.append(entry_node)
        entry_id = node_id
        node_id += 1
        
        # Walk the function body to extract control flow
        prev_node_id = entry_id
        node_stack = []  # Stack for handling nested structures
        
        def process_statement(stmt_cursor: Cursor, prev_id: int) -> int:
            nonlocal node_id
            
            if stmt_cursor.kind == CursorKind.IF_STMT:
                # Extract condition
                cond_text = ""
                children_list = list(stmt_cursor.get_children())
                
                # First child is typically the condition
                if children_list:
                    cond_cursor = children_list[0]
                    try:
                        tokens = [token.spelling for token in cond_cursor.get_tokens()]
                        cond_text = ' '.join(tokens[:20])  # Limit length
                        if not cond_text.strip():
                            cond_text = "condition"
                    except:
                        cond_text = "condition"
                
                if not cond_text:
                    cond_text = "condition"
                
                # Create condition node
                cond_node = CFGNode(id=node_id, type="condition", expr=cond_text)
                cfg.nodes.append(cond_node)
                cfg.edges.append(CFGEdge(from_node=prev_id, to_node=node_id))
                cond_id = node_id
                node_id += 1
                
                # Process then branch (second child typically)
                then_id = node_id
                then_node = CFGNode(id=then_id, type="statement")
                cfg.nodes.append(then_node)
                cfg.edges.append(CFGEdge(from_node=cond_id, to_node=then_id, label="true"))
                node_id += 1
                
                # Process else branch if exists (third child)
                has_else = len(children_list) >= 3
                if has_else:
                    else_id = node_id
                    else_node = CFGNode(id=else_id, type="statement")
                    cfg.nodes.append(else_node)
                    cfg.edges.append(CFGEdge(from_node=cond_id, to_node=else_id, label="false"))
                    node_id += 1
                    return else_id  # Return else node as continuation
                else:
                    return then_id  # Return then node as continuation
                
            elif stmt_cursor.kind == CursorKind.CALL_EXPR:
                # Function call - get the callee name
                callee = stmt_cursor.spelling or "unknown"
                # If spelling is empty, try to get from referenced cursor
                if callee == "unknown":
                    try:
                        referenced = stmt_cursor.referenced
                        if referenced:
                            callee = referenced.spelling or "unknown"
                    except:
                        pass
                
                # Filter out operator calls and other system calls
                if callee.startswith("operator") or callee.startswith("__"):
                    # Still create node but mark it differently
                    stmt_node = CFGNode(id=node_id, type="statement")
                    cfg.nodes.append(stmt_node)
                    cfg.edges.append(CFGEdge(from_node=prev_id, to_node=node_id))
                    node_id += 1
                    return node_id - 1
                
                call_node = CFGNode(id=node_id, type="call", callee=callee)
                cfg.nodes.append(call_node)
                cfg.edges.append(CFGEdge(from_node=prev_id, to_node=node_id))
                node_id += 1
                return node_id - 1
                
            elif stmt_cursor.kind == CursorKind.RETURN_STMT:
                return_node = CFGNode(id=node_id, type="return")
                cfg.nodes.append(return_node)
                cfg.edges.append(CFGEdge(from_node=prev_id, to_node=node_id))
                node_id += 1
                return node_id - 1
            
            # Default: statement node
            stmt_node = CFGNode(id=node_id, type="statement")
            cfg.nodes.append(stmt_node)
            cfg.edges.append(CFGEdge(from_node=prev_id, to_node=node_id))
            node_id += 1
            return node_id - 1
        
        # Process function body - find compound statement
        body_found = False
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body_found = True
                # Process statements in the compound statement
                stmt_list = list(child.get_children())
                if not stmt_list:
                    # Empty function body
                    pass
                else:
                    for stmt in stmt_list:
                        try:
                            prev_node_id = process_statement(stmt, prev_node_id)
                        except Exception as e:
                            # If processing fails, create a statement node and continue
                            stmt_node = CFGNode(id=node_id, type="statement")
                            cfg.nodes.append(stmt_node)
                            cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
                            prev_node_id = node_id
                            node_id += 1
                break
        
        # If no compound statement found, function might have inline body
        if not body_found:
            # Try to process direct children as statements
            for child in cursor.get_children():
                if child.kind != CursorKind.PARM_DECL:  # Skip parameters
                    try:
                        prev_node_id = process_statement(child, prev_node_id)
                    except:
                        pass
        
        # Ensure return node exists
        has_return = any(n.type == "return" for n in cfg.nodes)
        if not has_return:
            return_node = CFGNode(id=node_id, type="return")
            cfg.nodes.append(return_node)
            cfg.edges.append(CFGEdge(from_node=prev_node_id, to_node=node_id))
        
        return cfg


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
                        # Add false branch node and edge
                        false_node = CFGNode(id=node_id, type="statement")
                        cfg.nodes.append(false_node)
                        cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=node_id, label="false"))
                        prev_node_id = node_id
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
                    # Add true branch node and edge
                    true_node = CFGNode(id=node_id, type="statement")
                    cfg.nodes.append(true_node)
                    cfg.edges.append(CFGEdge(from_node=top_cond['node_id'], to_node=node_id, label="true"))
                    prev_node_id = node_id
                    node_id += 1
                    # Pop condition after processing
                    condition_stack.pop()
        
        # Ensure all condition nodes have both branches
        for cond_info in condition_stack:
            if not any(e.from_node == cond_info['node_id'] and e.label == "true" for e in cfg.edges):
                # Create true branch node
                true_node = CFGNode(id=node_id, type="statement")
                cfg.nodes.append(true_node)
                cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=node_id, label="true"))
                node_id += 1
            if not any(e.from_node == cond_info['node_id'] and e.label == "false" for e in cfg.edges):
                # Create false branch node
                false_node = CFGNode(id=node_id, type="statement")
                cfg.nodes.append(false_node)
                cfg.edges.append(CFGEdge(from_node=cond_info['node_id'], to_node=node_id, label="false"))
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
        
        # Final cleanup: Remove any edges that reference non-existent nodes
        valid_node_ids = {node.id for node in cfg.nodes}
        valid_edges = []
        for edge in cfg.edges:
            if edge.from_node in valid_node_ids and edge.to_node in valid_node_ids:
                valid_edges.append(edge)
            else:
                # Skip invalid edge
                pass
        
        cfg.edges = valid_edges
        
        # Ensure all nodes with no incoming edges (except entry) connect from entry
        nodes_with_incoming = {e.to_node for e in cfg.edges}
        orphaned_nodes = [n.id for n in cfg.nodes if n.id != entry_id and n.id not in nodes_with_incoming]
        for orphan_id in orphaned_nodes:
            if not any(e.from_node == entry_id and e.to_node == orphan_id for e in cfg.edges):
                cfg.edges.insert(0, CFGEdge(from_node=entry_id, to_node=orphan_id))
        
        return cfg
    
    @staticmethod
    def build_all_cfgs_from_source(source_dir: str) -> List[CFG]:
        """
        Build CFGs for ALL functions in the source directory.
        
        Returns list of CFGs, one per function.
        """
        cfgs = []
        
        # Try Clang first
        if ClangIntegration.check_clang_available():
            try:
                functions = ClangIntegration.discover_all_functions(source_dir)
                
                if not functions:
                    print("  [WARNING] Clang found no functions")
                    return cfgs
                
                # Find source files
                source_files_map = {}
                cpp_files = []
                for ext in ['*.cpp', '*.cc', '*.cxx', '*.c++']:
                    cpp_files.extend(glob.glob(os.path.join(source_dir, ext)))
                    cpp_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
                
                for cpp_file in cpp_files:
                    basename = os.path.basename(cpp_file)
                    source_files_map[basename] = cpp_file
                
                # Build CFG for each function
                for func_info in functions:
                    source_file = source_files_map.get(func_info.source_file)
                    if not source_file:
                        # Try to find file by name
                        for cpp_file in cpp_files:
                            if func_info.source_file in cpp_file or os.path.basename(cpp_file) == func_info.source_file:
                                source_file = cpp_file
                                break
                    
                    if source_file:
                        cfg = ClangIntegration.build_cfg_from_clang(source_file, func_info.name)
                        if cfg:
                            cfgs.append(cfg)
                        else:
                            print(f"  [WARNING] Failed to build CFG for {func_info.name}")
                
                return cfgs
                
            except Exception as e:
                print(f"  [ERROR] Clang processing failed: {e}")
                raise  # Fail clearly, don't fallback
        
        # If we reach here, Clang is not available
        raise RuntimeError(
            "Clang is REQUIRED but not available.\n"
            "Install Clang:\n"
            "  1. System: sudo apt-get install libclang-dev (Ubuntu) or brew install llvm (macOS)\n"
            "  2. Python: pip install clang\n"
            "  3. Set LIBCLANG_LIBRARY_PATH if needed\n"
        )
    
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
    """Build CFG, CallGraph, and Description for ALL functions in C++ source directory"""
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
    
    # Step 1.5: Get Agent7 repository root (where agent.py is located)
    AGENT_ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_ROOT = os.path.join(AGENT_ROOT, "output")
    
    print(f"[INFO] Agent root: {AGENT_ROOT}")
    print(f"[INFO] Output will be written to: {OUTPUT_ROOT}")
    
    # Step 2: Initialize Clang and discover ALL functions
    print("Step 2: Initializing Clang and discovering ALL functions...")
    
    if not ClangIntegration.check_clang_available():
        print("[ERROR] Clang is REQUIRED but not available.")
        print("Install Clang:")
        print("  1. System: sudo apt-get install libclang-dev (Ubuntu) or brew install llvm (macOS)")
        print("  2. Python: pip install clang")
        print("  3. Set LIBCLANG_LIBRARY_PATH if needed")
        sys.exit(1)
    
    # Load compile_commands.json first (needed for function discovery and CFG extraction)
    compile_commands_path = os.path.join(source_dir, 'compile_commands.json')
    if not os.path.exists(compile_commands_path):
        # Check parent directories (up to 3 levels)
        checked_paths = [compile_commands_path]
        current_dir = source_dir
        for _ in range(3):
            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            compile_commands_path = os.path.join(parent, 'compile_commands.json')
            checked_paths.append(compile_commands_path)
            if os.path.exists(compile_commands_path):
                break
            current_dir = parent
    
    if not os.path.exists(compile_commands_path):
        error_msg = (
            f"[ERROR] compile_commands.json is REQUIRED but not found.\n"
            f"  Searched in:\n"
        )
        for path in checked_paths[:5]:
            error_msg += f"    - {path}\n"
        error_msg += (
            f"\n  Please generate compile_commands.json:\n"
            f"    - cmake: cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .\n"
            f"    - bear: bear -- make\n"
            f"    - clang: clang -p .\n"
        )
        print(error_msg)
        sys.exit(1)
    
    # Load compile_commands.json
    compile_commands = None
    compile_args_map = {}
    try:
        with open(compile_commands_path, 'r') as f:
            compile_commands = json.load(f)
        print(f"  [OK] Found compile_commands.json: {compile_commands_path}")
        
        # Build file -> compile args mapping
        for entry in compile_commands:
            file_path = entry.get('file', '')
            if file_path:
                abs_file = os.path.abspath(file_path)
                args = entry.get('arguments', [])
                if not args:
                    # Parse command string if arguments not available
                    cmd = entry.get('command', '')
                    if cmd:
                        import shlex
                        args = shlex.split(cmd)
                        # Remove compiler name and output flags
                        args = [a for a in args[1:] if a not in ['-o', '-c'] and not a.endswith('.o')]
                compile_args_map[abs_file] = args
                compile_args_map[os.path.basename(file_path)] = args
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Failed to load compile_commands.json: {e}\n"
            f"  Path: {compile_commands_path}\n"
            f"  Please fix or regenerate compile_commands.json"
        )
    
    try:
        all_functions = ClangIntegration.discover_all_functions(source_dir, compile_args_map)
    except Exception as e:
        print(f"  [ERROR] Function discovery failed: {e}")
        sys.exit(1)
    
    if len(all_functions) == 0:
        print("  [ERROR] No functions found. Aborting.")
        sys.exit(1)
    
    if len(all_functions) < 5:
        print(f"  [WARNING] Only {len(all_functions)} function(s) detected. This may indicate a problem.")
        print(f"  [WARNING] Expected at least 5 functions. Check filtering and Clang parsing.")
    
    print(f"  [OK] Discovered {len(all_functions)} user-defined function(s)")
    
    # Step 3: Create output directory structure in Agent7 repository
    output_dir = OUTPUT_ROOT
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cfg"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "description"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "diagrams"), exist_ok=True)
    
    # Step 4: Process EACH function
    print(f"\nStep 4: Processing {len(all_functions)} function(s)...")
    
    # Validation: Ensure no system functions leaked through
    system_function_names = [f.name for f in all_functions 
                            if f.name.startswith("__") or f.name.startswith("operator")]
    if system_function_names:
        print(f"  [ERROR] System/compiler functions detected in results:")
        for name in system_function_names[:10]:
            print(f"    - {name}")
        print(f"  [ERROR] Aborting - filtering failed")
        sys.exit(1)
    
    registry = FunctionRegistry()
    all_cfgs = []
    global_callgraph = CallGraph()
    
    # Find source files ONLY within project directory
    project_root_abs = os.path.abspath(source_dir)
    source_files_map = {}
    cpp_files = []
    for ext in ['*.cpp', '*.cc', '*.cxx', '*.c++']:
        pattern = os.path.join(project_root_abs, ext)
        cpp_files.extend(glob.glob(pattern))
        pattern_recursive = os.path.join(project_root_abs, '**', ext)
        cpp_files.extend(glob.glob(pattern_recursive, recursive=True))
    
    # Filter to only files in project root
    cpp_files = [f for f in cpp_files if os.path.abspath(f).startswith(project_root_abs)]
    
    for cpp_file in cpp_files:
        basename = os.path.basename(cpp_file)
        source_files_map[basename] = cpp_file
    
    for func_info in all_functions:
        func_name = func_info.name
        print(f"\n  Processing function: {func_name}")
        
        # Find source file
        source_file = source_files_map.get(func_info.source_file)
        if not source_file:
            for cpp_file in cpp_files:
                if func_info.source_file in cpp_file or os.path.basename(cpp_file) == func_info.source_file:
                    source_file = cpp_file
                    break
        
        if not source_file:
            print(f"    [WARNING] Source file not found for {func_name}, skipping")
            continue
        
        # Build CFG using Clang (NO FALLBACK - must succeed)
        try:
            cfg = ClangIntegration.build_cfg_from_clang(source_file, func_name, compile_args_map)
            
            # Validate CFG was extracted correctly
            if not cfg:
                raise RuntimeError("CFG extraction returned None")
            if not cfg.nodes:
                raise RuntimeError("CFG has no nodes")
            if len(cfg.nodes) < 2:
                raise RuntimeError(f"CFG has only {len(cfg.nodes)} node(s), expected >= 2")
            
            print(f"    [OK] CFG extracted: {len(cfg.nodes)} nodes, {len(cfg.edges)} edges")
            
        except Exception as e:
            print(f"    [ERROR] Clang CFG extraction FAILED for {func_name}:")
            print(f"      {e}")
            print(f"    [ERROR] Aborting - CFG extraction is required for all functions")
            sys.exit(1)
        
        # Validate CFG
        cfg_validator = CFGValidationAgent()
        is_valid, issues = cfg_validator.validate(cfg)
        
        if not is_valid:
            print(f"    [ERROR] CFG validation FAILED for {func_name}:")
            for issue in issues:
                print(f"      - {issue}")
            print(f"    Aborting due to CFG validation failure.")
            sys.exit(1)
        
        all_cfgs.append(cfg)
        func_info.cfg = cfg
        
        # Build CallGraph for this function
        func_callgraph = CallGraphBuilder.build_from_cfg(cfg)
        if cfg.function:
            global_callgraph.calls[cfg.function] = func_callgraph.calls.get(cfg.function, [])
        
        # Build Description
        desc = DescriptionBuilder.build_from_cfg(cfg, global_callgraph)
        
        # Validate Description
        desc_validator = DescriptionValidationAgent()
        is_valid, corrected_desc, justification = desc_validator.validate(desc, cfg, global_callgraph)
        if not is_valid:
            desc = corrected_desc
        
        func_info.description = desc
        
        # Generate Mermaid
        primary_agent = PrimaryAnalysisAgent()
        mermaid = primary_agent.generate_mermaid(cfg, desc)
        
        # Validate Mermaid
        diagram_validator = DiagramValidationAgent()
        is_valid, corrected_mermaid, issues = diagram_validator.validate(mermaid, cfg)
        if not is_valid:
            mermaid = corrected_mermaid
        
        # Save artifacts for this function
        if not dry_run:
            # Create safe filename (should never have operators since they're filtered, but be safe)
            safe_func_name = func_name.replace(':', '_').replace('/', '_').replace('\\', '_').replace('<', '_').replace('>', '_').replace('*', '_').replace('&', '_').replace(' ', '_').replace('"', '_').replace("'", '_')
            
            # Final safety check - abort if somehow an operator got through
            if safe_func_name.startswith("operator"):
                print(f"    [ERROR] Operator overload detected in filename: {safe_func_name}")
                print(f"    [ERROR] Aborting - filtering validation failed")
                sys.exit(1)
            
            # Save CFG
            cfg_path = os.path.join(output_dir, "cfg", f"{safe_func_name}.json")
            CFGBuilder.save_cfg(cfg, cfg_path)
            
            # Save Description
            desc_path = os.path.join(output_dir, "description", f"{safe_func_name}.json")
            DescriptionBuilder.save_description(desc, desc_path)
            
            # Save Mermaid
            mermaid_path = os.path.join(output_dir, "diagrams", f"{safe_func_name}.mermaid")
            with open(mermaid_path, 'w') as f:
                f.write(mermaid + "\n")
            
            print(f"    [OK] Saved artifacts for {func_name}")
        
        registry.add_function(func_info)
    
    # Step 5: Validation - ensure all functions were processed
    print(f"\nStep 5: Validating function processing...")
    
    if len(all_cfgs) != len(all_functions):
        print(f"  [ERROR] CFG count mismatch!")
        print(f"    Discovered functions: {len(all_functions)}")
        print(f"    Generated CFGs: {len(all_cfgs)}")
        print(f"  [ERROR] Aborting - not all functions were processed")
        sys.exit(1)
    
    if len(all_cfgs) < 5:
        print(f"  [WARNING] Only {len(all_cfgs)} function(s) processed")
        print(f"  [WARNING] Expected at least 5 functions. This may indicate:")
        print(f"    - Filtering too strict")
        print(f"    - Clang parsing issues")
        print(f"    - Project has very few functions")
    
    print(f"  [OK] All {len(all_cfgs)} function(s) processed successfully")
    
    # Step 6: Build global CallGraph
    print(f"\nStep 6: Building global CallGraph...")
    
    # Aggregate all calls from all CFGs
    for cfg in all_cfgs:
        if cfg.function not in global_callgraph.calls:
            global_callgraph.calls[cfg.function] = []
        for node in cfg.nodes:
            if node.type == "call" and node.callee:
                if node.callee not in global_callgraph.calls[cfg.function]:
                    global_callgraph.calls[cfg.function].append(node.callee)
    
    print(f"  [OK] Global CallGraph built with {len(global_callgraph.calls)} function(s)")
    
    # Step 7: Validate global CallGraph
    print("\nStep 7: Validating global CallGraph...")
    cg_validator = CallGraphValidationAgent()
    all_valid = True
    for cfg in all_cfgs:
        is_valid, issues = cg_validator.validate(global_callgraph, cfg)
        if not is_valid:
            print(f"  [WARNING] CallGraph validation issues for {cfg.function}:")
            for issue in issues:
                print(f"    - {issue}")
            all_valid = False
    
    if all_valid:
        print("  [OK] Global CallGraph validation PASSED")
    
    # Save global CallGraph
    if not dry_run:
        cg_output = os.path.join(output_dir, "callgraph.json")
        CallGraphBuilder.save_callgraph(global_callgraph, cg_output)
        print(f"  [OK] Saved global CallGraph to: {cg_output}")
    
    # Step 7: Final validation and summary
    print("\n" + "=" * 60)
    print("=== PROCESSING COMPLETE ===")
    print("=" * 60)
    
    # Final validation: Ensure output is in Agent7 repository, not project path
    output_dir_abs = os.path.abspath(output_dir)
    agent_root_abs = os.path.abspath(AGENT_ROOT)
    
    if not output_dir_abs.startswith(agent_root_abs):
        print(f"\n[ERROR] Output directory validation FAILED!")
        print(f"  Output: {output_dir_abs}")
        print(f"  Agent root: {agent_root_abs}")
        print(f"  Output must be inside Agent7 repository")
        sys.exit(1)
    
    # Final validation: No operator overloads or system functions
    final_system_funcs = [f.name for f in all_functions 
                         if f.name.startswith("__") or f.name.startswith("operator")]
    if final_system_funcs:
        print(f"\n[ERROR] Final validation FAILED - system functions detected:")
        for name in final_system_funcs[:10]:
            print(f"  - {name}")
        sys.exit(1)
    
    print(f"\nProcessed {len(all_functions)} user-defined function(s):")
    for func_info in all_functions:
        if func_info.cfg:
            print(f"  - {func_info.name}: {len(func_info.cfg.nodes)} nodes, {len(func_info.cfg.edges)} edges")
    
    if not dry_run:
        print(f"\n=== OUTPUT STRUCTURE ===")
        print(f"Agent7/output/")
        print(f" ├── cfg/ ({len(all_functions)} files)")
        print(f" ├── description/ ({len(all_functions)} files)")
        print(f" ├── diagrams/ ({len(all_functions)} files)")
        print(f" └── callgraph.json")
        print(f"\nAll artifacts saved to: {output_dir}")
        print(f"  (Inside Agent7 repository, not project path)")


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

