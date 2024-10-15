import yaml
import logging

class Trainer:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        logging.basicConfig(level=logging.INFO)

    def add_ddl(self, ddl: str):
        """Add a DDL (Data Definition Language) statement directly into the vector store."""
        logging.info(f"Adding DDL: {ddl}")
        self.vector_store.add_sample(f"DDL: {ddl}")

    def add_documentation(self, documentation: str):
        """Add documentation directly into the vector store."""
        logging.info(f"Adding documentation: {documentation}")
        self.vector_store.add_sample(f"Documentation: {documentation}")

    def add_question_sql(self, question: str, sql_query: str):
        """Add a question and SQL pair directly into the vector store."""
        combined_sample = f"Question: {question}\nSQL: {sql_query}"
        logging.info(f"Adding question and SQL: {question}")
        self.vector_store.add_sample(combined_sample)

    def load_from_yaml(self, yaml_path: str):
        """Load training data from a YAML configuration file and insert directly into the vector store."""
        self.config = self._load_yaml(yaml_path)
        self.process_training_config()

    def process_training_config(self):
        """Process the loaded YAML training configuration."""
        if not self.config:
            raise ValueError("No configuration loaded. Provide a YAML path or config.")
        
        for item in self.config.get('training_data', []):
            if 'ddl' in item:
                self.add_ddl(item['ddl'])
            elif 'documentation' in item:
                self.add_documentation(item['documentation'])
            elif 'question' in item and 'sql' in item:
                self.add_question_sql(item['question'], item['sql'])
            else:
                logging.warning("Unrecognized entry in training data: %s", item)

        logging.info("Training data loaded into the vector store.")
    
    def _load_yaml(self, yaml_path):
        """Load YAML data from a file."""
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)


