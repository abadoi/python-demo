from __future__ import print_function
from ctypes import CFUNCTYPE, c_double, c_long
import llvmlite.binding as llvm
import sys
import os
import pprint
import re
from difflib import SequenceMatcher
from collections import Counter 
from operator import itemgetter
from llvmlite import ir
from graphviz import Digraph

def main():
    if len(sys.argv) < 3:
        print("Error: 2 arguments required: <filename1> <filename2> . Exiting...")
        sys.exit()

    for filepath in sys.argv:
        if not os.path.isfile(filepath):
            print("File path {} does not exist. Exiting...".format(filepath))
            sys.exit()
    
    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]
    #TODO: add vulnerable line as arg/fuzztastic output
    input_lines = [7, 8, 9, 10, 11, 12]

    clang_ir = open(filepath1).read()
    retdec_ir = open(filepath2).read()

    mod_clang = compile_ir(clang_ir)
    mod_retdec = compile_ir(retdec_ir)
    parse_module(mod_clang)

    source_lines_map = {}
    for source_line in input_lines:
        list_of_dbg_instr = get_list_of_debug_indexes(clang_ir, source_line)
        source_lines_map.update({source_line : []})
        
        for index, item in list_of_dbg_instr:
            found = line_lookup(mod_clang, index)
            source_lines_map[source_line].append(found)
            print("Debug index: ", index, item)
            # pprint.pprint(found)
    
    pprint.pprint(source_lines_map)

    #TODO: look for lines in retdec that are similar/have the same prop as source_lines_map


def compile_ir(llvm_ir):
    """
    Compile the LLVM IR string with the given engine.
    The compiled module object is returned.
    """
    # Create a LLVM module object from the IR
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()

    return mod


def get_list_of_debug_indexes(llvm_ir, source_line):
    line_metadata = []
    for line in llvm_ir.splitlines():
        if re.search(r'^!\d+', line):     
                r = re.search(r'\bline: \d+', line.rstrip())
                line_number = int(r.group(0).split()[1]) if r else ""
                line_metadata.append({ 
                        'metadata': line.rstrip(),
                        'source_line': line_number
                })

    list_debug_indexes = [(i, item) for i, item in enumerate(line_metadata) if item['source_line'] == source_line]

    return list_debug_indexes

def line_lookup(m, dbg):
    result = {}
    for f in m.functions:
        for b in f.blocks:
            for i in b.instructions:
                debug_line = re.findall (r'(?<!\w)!\d+', str(i))
                is_debug_line = re.findall (r'(?<!\w)@llvm\.dbg\.\w+', str(i))
                # print(dbg, is_debug_line, debug_line, str(i))
                if not is_debug_line and debug_line and int(debug_line[0].lstrip('!')) == dbg:
                    result.update({
                        'function' : f.name,
                        'function_arguments' : [str(a.type) for a in f.arguments],
                        'operation' : i.opcode,
                        'operands' : [str(o.type) for o in i.operands],
                        'instruction' : str(i)
                    })

    return result

def parse_module(m):
    print(f'Parsing module: {m.name}')
    for f in m.functions:
        print(f'Function: {f.name}/`{f.type}`/ {f}')
        print(f'Function attributes: {list(f.attributes)}')
        for a in f.arguments:
            print(f'Argument: {a.name}/`{a.type}`')
            print(f'Argument attributes: {list(a.attributes)}')
        for b in f.blocks:
            print(f'Block: {b.name}/`{b.type}`\n {b} \nEnd of Block')
            for i in b.instructions:
                print(f'Instruction: {i.name}/`{i.opcode}`/`{i.type}`: `{i}`')
                print(f'Attributes: {list(i.attributes)}')
                for o in i.operands:
                    print(f'Operand: {o.name}/{o.type}')


if __name__ == '__main__':
    main() 