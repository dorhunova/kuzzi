# kuzzi
# kuzzi

## Overview
`kuzzi` is a Python-based project in progress aimed at transforming natural language queries into SQL queries. 

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/dorhunova/kuzzi.git
    cd kuzzi
    ```

2. **Set up environment variables:**
    Create a `.env` file in the root directory and add the necessary environment variables.
    ```env
    # Postgres Config
    DB_HOST=your_postgres_host
    DB_NAME=your_postgres_db_name
    DB_USER=your_postgres_user
    DB_PASSWORD=your_postgres_password
    DB_PORT=your_postgres_port

    # Azure OpenAI Chat Config
    AZURE_CHAT_API_BASE=your_azure_chat_api_base
    AZURE_CHAT_API_VERSION=your_azure_chat_api_version
    AZURE_CHAT_API_KEY=your_azure_chat_api_key
    AZURE_CHAT_DEPLOYMENT=your_azure_chat_deployment

    # Azure OpenAI Embeddings Config
    AZURE_EMBEDDINGS_API_BASE=your_azure_embeddings_api_base
    AZURE_EMBEDDINGS_API_VERSION=your_azure_embeddings_api_version
    AZURE_EMBEDDINGS_API_KEY=your_azure_embeddings_api_key
    AZURE_EMBEDDINGS_DEPLOYMENT=your_azure_embeddings_deployment

    # AWS Bedrock Config
    AWS_DEFAULT_REGION=your_aws_default_region
    AWS_BEDROCK_MODEL=your_aws_bedrock_model
    ```

3. **Build and run the Docker container:**
    ```sh
    docker-compose up --build
    ```

4. Connect your AWS credentials by ensuring the folder specified in the `docker-compose.yaml` file is correct: `~/.aws:/root/.aws:ro`. 
   Update the profile name in the `docker-compose.yaml` file under the `environment` section to match your AWS credentials profile. Specifically, change the value of `AWS_PROFILE` to your AWS profile name.

## Usage

### Running the Application
The application can be run using Docker. It will automatically start the main script defined in `kuzzi/main.py`.

