#!/usr/bin/env python3
"""
Convert all uppercase 'G' to lowercase 'g' in tlLogic phase state attributes.
This ensures that all green phases are "safe" (g) instead of "unsafe" (G).
"""

import xml.etree.ElementTree as ET
import sys
import os

def convert_g_to_g_in_phases(xml_file, output_file=None):
    """
    Read the XML file, find all <phase> elements inside <tlLogic>,
    and replace 'G' with 'g' in their 'state' attribute.

    Args:
        xml_file (str): Path to input .net.xml file.
        output_file (str, optional): Path to output file. If not provided,
                                     the input file is overwritten.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate over all tlLogic elements
    for tllogic in root.findall('tlLogic'):
        # Iterate over all phase elements inside the current tlLogic
        for phase in tllogic.findall('phase'):
            if 'state' in phase.attrib:
                state = phase.attrib['state']
                # Replace all 'G' with 'g'
                new_state = state.replace('G', 'g')
                if new_state != state:
                    phase.attrib['state'] = new_state
                    print(f"Modified phase in tlLogic '{tllogic.get('id')}': {state} -> {new_state}")

    # Write back the modified XML
    if output_file is None:
        output_file = xml_file
    tree.write(output_file, encoding='UTF-8', xml_declaration=True)
    print(f"Processed file saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py input.net.xml [output.net.xml]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Check if input file exists
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    convert_g_to_g_in_phases(input_path, output_path)