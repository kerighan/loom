"""
Template system for nested data structures.

Allows creating data structures that contain other data structures,
e.g., List of Lists, Dict of Lists, etc.
"""


class DataStructureTemplate:
    """Template for creating nested data structures.

    A template encapsulates:
    - The data structure class (List, Dict, etc.)
    - The dataset to use for storage
    - Configuration parameters (growth_factor, cache_size, etc.)

    Templates enable nested structures like List[List[User]].

    Example:
        # Create template
        UserList = List.template(user_dataset, cache_size=10)

        # Create list of lists
        teams = db.create_list('teams', UserList)

        # Append creates nested lists
        eng = teams.append()
        eng.append({'id': 1, 'name': 'Alice'})
    """

    def __init__(self, ds_class, dataset, config):
        """Initialize template.

        Args:
            ds_class: DataStructure class (List, Dict, etc.)
            dataset: Dataset for storage
            config: Configuration dict
        """
        self.ds_class = ds_class
        self.dataset = dataset
        self.config = config
        self._counter = 0  # For auto-generating names

    def new(self, db, name=None, **kwargs):
        """Create new instance from this template.

        Args:
            db: Database instance
            name: Optional name (auto-generated if None)
            **kwargs: Additional arguments to pass to constructor (e.g., _parent)

        Returns:
            New DataStructure instance
        """
        if name is None:
            # Auto-generate unique name
            name = f"_{self.ds_class.__name__}_{id(self)}_{self._counter}"
            self._counter += 1

        # Merge config with kwargs (kwargs take precedence)
        all_config = {**self.config, **kwargs}
        return self.ds_class(name, db, self.dataset, **all_config)

    def get_ref_schema(self):
        """Get schema for storing references to instances of this template.

        Returns:
            Dict mapping field names to numpy dtypes
        """
        # Minimal base fields
        schema = {
            "valid": "bool",  # 1 byte - for soft deletion
        }

        # Add DS-specific fields (includes metadata for nested structures)
        schema.update(self.ds_class._get_nested_ref_schema())

        return schema

    def __repr__(self):
        return f"Template({self.ds_class.__name__}[{self.dataset.name}])"
