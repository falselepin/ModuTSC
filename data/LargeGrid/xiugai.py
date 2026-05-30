import xml.etree.ElementTree as ET
tree = ET.parse('tripinfo.xml')
arrived = len(tree.getroot().findall('tripinfo'))
print(f"Arrived vehicles: {arrived}")
print(f"Throughput: {arrived / 3600:.2f} veh/s")