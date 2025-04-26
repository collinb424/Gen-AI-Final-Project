## Quick Start

1. **Clone this repository**  
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. **Set your API keys**  
   - In `embeddings.py` and `llm.py`, make sure you set _API_KEY with a valid key

3. **Upload PDFs**
   - In src/ create a directory entitled data/ and upload any pdfs there that you would like the model to answer questions based on

4. **Build and Start the Docker Containers**  
    ```bash
     docker-compose build
     ```
     ```bash
     docker-compose up -d
     ```

5. **Enter the Docker container**  
   ```bash
   docker exec -it final-project-anaconda-1 bash
   ```

6. **Index the documents**  
   Inside the container:
   ```bash
   cd code/
   python3 src/index.py
   ```
   This will load and embed your documents into the Milvus vector database.

7. **Start the Chainlit app**  
   Inside the container:
   ```bash
   chainlit run src/quote.py --host 0.0.0.0 --port 8000
   ```
   This will launch the UI at `http://localhost:8000`.

8. **Interact with the system!**
   - Ask questions.
   - Retrieve answers with citations and quotes!
