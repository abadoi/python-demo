import subprocess
import locale
import os
import sys
from os import listdir
from os.path import isfile, join
import json
import pprint
import pandas as pd
import numpy as np
from enum import Enum
import re
from os import path
from typing import List


class CWE(Enum):
    # everything else not vulnerable = 0
    _nofault = 0
    _284 = 1 
    _435 = 2
    _664 = 3
    _682 = 4
    _691 = 5
    _693 = 6
    _697 = 7
    _703 = 8
    _707 = 9
    _710 = 10

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def find_file(parent, endswith):
    for file_name in os.listdir(parent):
        if file_name.endswith(endswith):
            return os.path.join(parent, file_name)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                find_file(current_path, endswith)
    return None

def find_files(source_dir: str, exclude_dirs: List[str] = [], file_exts: List[str] = ['h', 'c']) -> List[str]:
    """
    This function returns the absolute paths of files from the base directory that match the file extension and are not
    in the excluded subdirectories.
    """
    source_files = []
    pattern = re.compile("^.*\.({0})$".format("|".join(file_exts)))

    for root, _, files in os.walk(path.expanduser(source_dir)):
        if root not in exclude_dirs:
            for file in filter(lambda f: pattern.match(f), files):
                source_files.append(path.join(root, file))

    return source_files


def get_llvm_ir(bitcode_file):
    try:
        command = ("extract-bc", bitcode_file)
        subprocess.run(command, encoding=locale.getpreferredencoding())
    except:
        print("extract-bc command not found.")

def get_binary_llvm(c_src_file, llvm_ir_file):
    command = (CLANG, '-S', '-emit-llvm', '-g', '-O0', c_src_file,  '-o', llvm_ir_file)
    subprocess.run(command, encoding=locale.getpreferredencoding())

def get_retdec(binary_file, file_name):
    try:
        command = ('retdec-decompiler', binary_file, '-o', file_name, "--no-memory-limit")
        subprocess.run(command, encoding=locale.getpreferredencoding())
    except:
        print("Retdec not available.")

# this method is used to match instructions to basic blocks based on source line and function name
def merge_files(instructions, basic_blocks, pprint = False):
    for bb in basic_blocks:
        lines = bb['lines']
        if 'fault_instructions' not in bb:
            fault_instructions = {}
        else:
            fault_instructions = bb['fault_instructions']

        for line in lines:
            for instr in instructions:
                if instr["source_line"] == line and instr["function"] == bb["function"]:
                    if line in fault_instructions:
                        fault_instructions[line].append(instr)
                    else:
                        fault_instructions[line] = [instr]
        bb.update({'fault_instructions': fault_instructions})

    if pprint:
        pprint.pprint(basic_blocks)

    return basic_blocks

def merge_files_byId(instructions, basic_blocks, pprint = False):
    for bb in basic_blocks:
        if 'instructions' not in bb:
            bb_instructions = []
        else:
            bb_instructions = bb['instructions']    
        for instr in instructions:
            if instr["bbId"] == bb["id"] and instr["function"] == bb["function"]:
                bb_instructions.append(instr)
        bb.update({'instructions': bb_instructions})

    if pprint:
        pprint.pprint(basic_blocks)

    return basic_blocks

def load_data(instructions, basic_blocks, unlabeled = False, pprint = False):
    out_dir = os.path.dirname(instructions)
    merged_data = []
    if unlabeled:
        merged_data_file_name = 'unlabeled_merged_data.json'
    else:
        merged_data_file_name = 'merged_data.json'

    merged_data_file = find_file(out_dir, merged_data_file_name)

    if merged_data_file:
        print("Loading existing merged file: {}".format(merged_data_file))
        with open(merged_data_file) as f:
            merged_data = json.load(f)
    else:
        print("Loading instructions...")
        instructions_data = None
        with open(instructions, 'r') as f:
            data = f.read()
            try:
                instructions_data = json.loads(data)
            except json.decoder.JSONDecodeError as e:
                print("Extra data... trying to merge batch file...")
                new_data = data.replace('}\n{', '},\n{')
                instructions_data_batched = json.loads(f'[{new_data}]')
                # merge batches
                instructions_data = {'retdec_instructions' : []}
                for batch in instructions_data_batched:
                    instructions_data['retdec_instructions'].extend(batch['retdec_instructions'])

        print("Done. \nLoading basic blocks...")
        bb_data = None
        with open(basic_blocks, 'r') as f:
            data = f.read()
            bb_data = json.loads(data)
    
        if pprint is True: 
            pprint.pprint(instructions_data)
            pprint.pprint(bb_data)
        
        print("Done.")

        if instructions_data and bb_data:
            print("Merging...")
            if unlabeled:
                merged_data = merge_files_byId(instructions_data['retdec_instructions'], bb_data['basic_blocks'])
            else:
                merged_data = merge_files(instructions_data['retdec_instructions'], bb_data['basic_blocks'])
            with open(os.path.join(out_dir, merged_data_file_name), 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=4)
        else:
            print("Could not load instructions and/or basic_blocks.")
        print("Done.")

    return merged_data


def create_df_rows(basic_blocks, binary = False):
    rows_list = []
    for bb in basic_blocks:
        if 'fault_instructions' in bb:
            for key, values in bb['fault_instructions'].items():
                d = {}      
                for value in values:
                    d.update(value['features'])
                    d.update({ "bb_length": bb["bb_length"], 
                            "pred_bb_num": bb["pred_bb_num"], 
                            "suc_bb_num": bb["suc_bb_num"]} )
                    if value['faults']:
                        #TODO: can have multiple faults per instruction
                        fault_name = '_' + value['faults'][0]['fault']['name'].split('-')[1] 
                        if binary:
                            d.update({'label': 1})
                        else:
                            d.update({'label': CWE[fault_name].value})
                    else:
                        d.update({'label': CWE._nofault.value})
                    # d.update({'instr_no': value['instr_no']})
                    rows_list.append(d)
        #unlabeled data
        elif 'instructions' in bb:
            d = {}      
            for instr in bb['instructions']:
                d.update(instr['features'])
                d.update({ "bb_length": bb["bb_length"], 
                        "pred_bb_num": bb["pred_bb_num"], 
                        "suc_bb_num": bb["suc_bb_num"]} )
                # d.update({'instr_no': value['instr_no']})
                rows_list.append(d)
    
    return rows_list              

def create_df_rows_from_instructions(instructions):
    rows_list = []
    for i in instructions["retdec_instructions"]:
        d = {}      
        d.update(i['features'])
        d.update({ "bb_length": i["bb_length"], 
                "pred_bb_num": i["pred_bb_num"], 
                "suc_bb_num": i["suc_bb_num"]} )
        d.update({'instr_no': i['instr_no']})
        rows_list.append(d)

    return rows_list              

def pattern_matching(pattern_matching_tool, fault_line_info, llvm_ir, retdec, out_dir):
    # '-s' flag to skip non-faulty instructions
    args = (pattern_matching_tool, '-f', fault_line_info, llvm_ir, retdec, out_dir, '-s')
    print("Executing pattern matching script with args: ", args)

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout, stderr = popen.communicate()
    status = popen.returncode
    print("Status: ", status)
    if not status:
        print (stdout.decode("utf-8"))
        
    return status


def parse_ir_file(pattern_matching_tool, retdec, out_dir):
    args = (pattern_matching_tool, retdec, out_dir)
    print("Parsing script with args: ", args) 

    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode()

    print (output)


def predict(pattern_matching_tool, ir_file, ml_model, out_dir):
    args = (pattern_matching_tool, ir_file, out_dir, '-m', ml_model)
    print("Executing pattern matching script with args: ", args)

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout, stderr = popen.communicate()
    status = popen.returncode
    print("Status: ", status)
    if not status:
        print (stdout.decode("utf-8"))
        
    return status

def evaluate(pattern_matching_tool, ir_file, retdec, fault_line_info, ml_model, out_dir):
    args = (pattern_matching_tool, '-f', fault_line_info, ir_file, retdec, '-m', ml_model, out_dir, '-s')
    print("Executing pattern matching script with args: ", args)

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    stdout, stderr = popen.communicate()
    status = popen.returncode
    print("Status: ", status)
    if not status:
        print (stdout.decode("utf-8"))
        
    return status

#filter data based on number of faults per basic block
#filter data based on number of different scanners per instr/bb
def filters(basic_blocks, filter_faults = None, filter_scanners_instr = None, filter_scanners_bb = None):
    print("Initial len {}".format(len(basic_blocks)))
    for bb in basic_blocks[:]:
        faults_per_bb = 0
        scanners_per_bb = []
        if 'fault_instructions' in bb:
            scanners_per_instr = 0
            for key, values in bb['fault_instructions'].items():   
                for value in values[:]:
                    faults_per_bb += len(value['faults'])
                    for fault in value['faults']:
                        scanners_per_instr = len(fault["found_by"])
                        scanners_per_bb.extend(fault["found_by"])
                    if filter_scanners_instr and scanners_per_instr < filter_scanners_instr:
                        print("deleting instr from bb with id ", bb["id"])
                        bb['fault_instructions'][key].remove(value)
        if filter_faults and faults_per_bb < filter_faults:
            print("deleting basic block with id ", bb["id"])
            basic_blocks.remove(bb)
        if filter_scanners_bb and len(set(np.unique(scanners_per_bb))) < filter_scanners_bb and bb in basic_blocks:
            print("deleting basic block with id ", bb["id"])
            basic_blocks.remove(bb)

    print("Final len {}".format(len(basic_blocks)))  
    return basic_blocks