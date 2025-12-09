# Nested Lists
# Lists of lists - for hierarchical data

from loom.database import DB
from loom.datastructures import List
import os

# Clean up any existing database
try:
    if os.path.exists("nested_example.loom"):
        os.remove("nested_example.loom")
except PermissionError:
    pass

db = DB("nested_example.loom")

# First, create a dataset for the inner list items
member_dataset = db.create_dataset("members", name="U50", role="U20")

# Create a template for nested lists
MemberListTemplate = List.template(member_dataset)

# Create outer list (list of lists)
teams = db.create_list("teams", MemberListTemplate)

# Append creates a new inner list
engineering = teams.append()  # Returns a new List
engineering.append({"name": "Alice", "role": "Lead"})
engineering.append({"name": "Bob", "role": "Senior"})

marketing = teams.append()
marketing.append({"name": "Carol", "role": "Manager"})

# Access nested lists
print(len(teams))           # 2 teams
print(len(teams[0]))        # 2 members in engineering
print(teams[0][0])          # {'name': 'Alice', 'role': 'Lead'}

# Iterate over nested structure
for i, team in enumerate(teams):
    print(f"Team {i}:")
    for member in team:
        print(f"  - {member['name']} ({member['role']})")

db.close()

# Cleanup
try:
    os.remove("nested_example.loom")
except PermissionError:
    pass  # File still in use on Windows
