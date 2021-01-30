import sys
import os
import pprint
import re
from difflib import SequenceMatcher
from collections import Counter 
from operator import itemgetter
from llvmlite import ir


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
    function_keywords = ["define", "call"]

    with open(filepath1) as fp:
        line_number = 0
        results = []
        for line in fp:
            line_number += 1
            if any(x in line.split() for x in function_keywords):
                operation = next((x for x in function_keywords if x in line.split()), False)
                function_name = re.findall (r'(?<!\w)@\w+', line)
                function_signature = re.findall (r'\((.*?)\)', line)
                number_of_operands = len(function_signature[0].split(','))
                bitcode_instr = {
                    "line_number" : line_number,
                    "retdec_instruction" : line.rstrip(),
                    "operation": operation,
                    "function_name": function_name,
                    "number_of_operands": number_of_operands
                }

                # best matches
                best_matches = find_all_matches(bitcode_instr, filepath2)

                # Take best match
                source_line = line_lookup(best_matches[0], filepath2)
                bitcode_instr.update({"source_line" : source_line})
                results.append(bitcode_instr)

    print("Result:\n")
    pprint.pprint(results)

def find_all_matches(instr, filepath):
    function_keywords = ["define", "call"]
    
    matches = []
    with open(filepath) as fp2:
        for line in fp2:
            s = similar (line.rstrip(), instr["retdec_instruction"])
            common_el = common_elements(re.split('[= ]', instr["retdec_instruction"]), line.split())
            if common_el > 0 and s > 0.5:
                #TODO: compare function names, number of operands
                debug = re.findall (r'(?<!\w)!\d+', line)
                matches.append({
                    "bitcode_instruction": line.rstrip(),
                    "similarity": s,
                    "common_elements": common_el,
                    "debug": debug

                })
    # sort based on similarity score 
    matches = sorted(matches, key=itemgetter('similarity'), reverse=True)

    # print("\nRetdec instruction:")
    # pprint.pprint(instr)

    # print("\Matched instructions:")
    # pprint.pprint(matches)

    return matches

def line_lookup(match, filepath): 
    with open(filepath) as fp3:
        for line in fp3:
            if re.search(r'^!\d+', line):     
                debug = re.findall(r'^!\d+', line)
                if debug[0] == match["debug"][0]:
                    r = re.search(r'\bline: \d+', line.rstrip())
                    # print("M", r)
                    result = r.group(0) if r else ""
                    line_number = result.split()[1]
                    return line_number


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def common_elements(a, b): 
    a_set = set(a) 
    b_set = set(b) 
  
    if (a_set & b_set): 
        # print(a_set & b_set) 
        return len(a_set & b_set)
    else: 
        # print("No common elements") 
        return 0


if __name__ == '__main__':
    main() 