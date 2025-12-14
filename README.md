# Team_Rekommender3_Capstone_Project

## Team Members
Archana Shivashankar, Zach Xie, Akshata Madavi 

---
# Job Recommendation AI Assistant

An end-to-end pipeline that scrapes internship and new grad job postings from Simplify's GitHub listings, builds a vector database of jobs, and exposes an interactive AI assistant that reads a candidate's resume (PDF) and recommends matching roles.

## Project Links

| Resource | Link |
|----------|------|
| **Slides** | https://docs.google.com/presentation/d/1R31J4VawXAj1IaToMdOcQRuHf0GGLQK4mkfS4TO7Ydg/edit?slide=id.p#slide=id.p |
| **Demo Video** |https://www.youtube.com/watch?v=rhZ1ePQzmBs|
| **Report** |https://docs.google.com/document/d/14sGbY7jrOywQAhMyLkHuN_EKH7GRcULr03lqQThP8eU/edit?tab=t.0 |

---

## Project Overview

This project automates three major steps in the early job search workflow:

- Scrape structured job data (internship and new grad) from a Simplify GitHub page using a custom web crawler.
- Build a semantic job matching engine using sentence embeddings, engineered numeric/categorical features, and a vector database (ChromaDB).
- Provide a Gradio based AI assistant that:
  - Extracts a candidate profile from a PDF resume using Gemini.
  - Fills in missing fields via an interactive chat.
  - Returns a ranked list of recommended jobs with links and metadata.

## Architecture

<img width="1327" height="697" alt="Screenshot 2025-12-09 at 4 48 48 PM" src="https://github.com/user-attachments/assets/2c540cac-9db1-4e57-9ac5-ed7ceac604fe" />

| Layer              | Description                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|
| Data collection    | Web crawler hits Simplify's GitHub job listings (intern + new grad) and writes a CSV.       |
| Feature pipeline   | Cleans and transforms job and candidate fields, then builds composite feature vectors.       |
| Vector database    | Stores job vectors plus rich metadata in a persistent ChromaDB collection.                  |
| Matching logic     | Filters jobs by eligibility (degree, sponsorship, job type) and ranks by cosine similarity. |
| AI assistant (UI)  | Gradio app + Gemini for resume parsing, profile completion, and job recommendation display. |

## 1. Data Collection (Web Crawler + CSV)

A separate script (run before this notebook) crawls the Simplify jobs GitHub page to extract internship and new grad roles and writes them to a CSV file. The resulting CSV is the input to this notebook.

**Output CSV schema (per job):**

- `job_name` / `Role`: Job title or role name.
- `company`: Company name.
- `skills_required` / `skill_sets`: Skills or tags associated with the job.
- `gives_sponsorship` / `Provide_Sponsorship`: Boolean indicating whether the job provides visa sponsorship.
- `url`: Direct link to the job posting.
- `education` / `Diploma`: Minimum degree requirement text.
- `job_type`: e.g., `Intern`, `Full-Time` (intern/new grad classification).

The notebook expects a cleaned job CSV (e.g., `jobs_df_demo.csv`) located in Google Drive under the configured path.

## 2. JobsRecommendation Notebook

The `JobsRecommendation.ipynb` notebook is responsible for:

### Environment Setup

- Installs dependencies: `chromadb`, `sentence-transformers`, `gradio`, `google-generativeai`, etc.
- Mounts Google Drive and sets a persistent ChromaDB path (e.g., `/content/drive/MyDrive/demoDB`).
- Loads the SentenceTransformers model `all-MiniLM-L6-v2` for text embeddings.

### Feature Engineering

The notebook builds a feature pipeline that combines structured job fields with text embeddings:

- **Degree normalization (`resolve_degree_rank`)**
  - Maps free form education strings to a numeric rank:
    - 0 = no explicit degree / high school
    - 1 = Bachelor
    - 2 = Master
    - 3 = PhD

- **Thermometer encoding for degrees (`ThermometerEncoder`)**
  - Encodes degree rank as a cumulative binary vector, e.g.:
    - Rank 1 = `[1, 0, 0]`
    - Rank 2 = `[1, 1, 0]`
    - Rank 3 = `[1, 1, 1]`

- **FeatureVectorization**
  - Uses a `ColumnTransformer` with:
    - `YOE` (years of experience) scaled via `MinMaxScaler`.
    - `Diploma` encoded via `ThermometerEncoder`.
    - Other fields (job type, sponsorship flags, etc.) are dropped from the similarity vector but retained as metadata.
  - Computes text embeddings over `skill_sets` using `all-MiniLM-L6-v2`.
  - Concatenates structured features + skill embeddings into a composite vector per job.

### Job Data Loading and Cleaning

- Loads the jobs CSV (e.g., `jobs_df_demo.csv`) into `jobs_df`.
- Ensures `Job_type` is normalized:
  - If `Role` contains "intern" (caseinsensitive), label as `Intern`, else `Full-Time`.
- Computes `degree_rank` from the `Diploma` column using `resolve_degree_rank`.
- (Optional) Saves back to CSV for reuse.

### Vector Database (ChromaDB)

- Initializes a persistent ChromaDB client pointing to `DB_PATH`.
- Creates or loads a collection (e.g., `job_postings_demo`) with cosine similarity as the metric.
- If the collection is empty:
  - Runs the `FeatureVectorization` pipeline on `jobs_df` to generate job vectors.
  - Builds metadata records for each job:
    - `Company`, `Role`, `YOE`, `Diploma`, `Job_type`, `skill_sets`, `Provide_Sponsorship`, `url`, `min_degree_req` (numeric).
  - Inserts:
    - `embeddings`: composite job vectors.
    - `documents`: text representation (`skill_sets`).
    - `metadatas`: full metadata dict.
    - `ids`: stringified row indices.
- Refits the `FeatureVectorization` pipeline to support candidate vectorization.

### Matching Logic

There are two entry points for matching:

1. **Batch mode (`match_jobs_from_csv`)**
   - Accepts a `candidates_df` containing one or more candidate rows.
   - For each candidate, constructs a `candidate_profile` with fields:
     - `YOE`, `Diploma`, `skill_sets`, `Job_type`, `Require_Sponsorship`.
   - Builds filters:
     - Degree filter: jobs with `min_degree_req <= candidate_degree_rank`.
     - Sponsorship filter:
       - If candidate requires sponsorship only jobs where `Provide_Sponsorship == True`.
       - If not â†’ no restriction.
     - Job type filter: matches candidate's `Job_type` preference (intern/fullâ€‘time).
   - Combines filters into a ChromaDB `where` clause using `$and`.
   - Converts the candidate profile into a vector using `vectorizer.transform_candidate`.
   - Queries ChromaDB with `n_results=5` and the combined filters.
   - Prints and returns a markdown table summarizing top matches:
     - `Candidate_Index`, `Company`, `Role`, similarity `Score`, `Job Type`, `Provide_Sponsorship`, `Required Skills`, `URL`.

2. **Single JSON profile (`match_jobs`, `process_json_to_dataframe`)**
   - `process_json_to_dataframe` converts a candidate JSON object into a oneâ€‘row DataFrame with columns:
     - `YOE`: numeric years of experience (string "NULL" handled as NaN).
     - `Diploma`: current degree and major.
     - `Job_type`: job preference (intern / full time / both / NULL).
     - `Require_Sponsorship`: boolean derived from `"require_sponsorship"` ("Yes" `True`).
     - `skill_sets`: merged, deduplicated list of `programming_languages` and `tools_frameworks`.
   - `match_jobs` calls `match_jobs_from_csv` on this DataFrame and returns the markdown table string.

## 3. AI Assistant UI (Gradio + Gemini)

The notebook exposes an interactive UI where candidates upload resumes and chat with an agent to finalize their profile and get recommendations.

### Gemini Integration

- Uses `google-generativeai` (Gemini 2.5 Flash) for:
  - Uploading the resume PDF.
  - Extracting a structured JSON candidate profile with keys like:
    - `graduation_date`, `current_degree_major`, `current_degree_gpa`, `require_sponsorship`,
      `programming_languages`, `tools_frameworks`, `leadership`, `job_preference`, `impact_outcomes`.
  - Enforces strict rules in the prompt to output `"NULL"` for missing fields.

### Stateful Agent (`ResumeChatBot`)

- Manages conversation state:
  - `INIT`, `MISSING_KEY`, `WAITING_FOR_RESUME`, `VERIFYING`, `COMPLETE`.
- Stores:
  - `extracted_data`: the JSON profile from Gemini.
  - `missing_queue`: list of fields that need interactive clarification.
  - `current_missing_key`: the field currently being asked about.
- On file upload:
  - Sends the PDF to Gemini and waits until the file is processed.
  - Parses the JSON response into `extracted_data`.
  - Identifies missing or `"NULL"` fields based on `FIELD_CONFIG`, prioritizing:
    - Interactive fields (graduation date, degree, GPA, sponsorship, YOE, job preference).
    - "Fallback" case when the resume is very sparse (many NULLs).
  - If nothing is missing, calls `finalize_report` immediately.
  - Otherwise transitions to `VERIFYING` and returns the first question.

- `next_question`:
  - Pops the next field from `missing_queue`.
  - Returns a natural language question using field metadata from `FIELD_CONFIG`.

- `handle_chat`:
  - In verification state, records the user's answer for `current_missing_key` and asks the next question.
  - In complete state, informs the candidate that the profile is done and suggests uploading a new resume to restart.
  - If waiting for resume, prompts the user to upload a file first.

- `finalize_report`:
  - Calls `match_jobs(self.extracted_data)` to get job recommendations.
  - Saves the final candidate profile to `final_candidate_profile.json`.
  - Returns the markdown table containing top job matches.

### Gradio Frontend

- Built with `gr.Blocks` and organized into:
  - **Left panel**:
    - `gr.Chatbot` showing the conversation (intro message + Q&A).
    - `gr.Textbox` for user responses to followâ€‘up questions.
  - **Right panel**:
    - `gr.File` input to upload a PDF resume.
  - **Status area**:
    - A `gr.Markdown` box displaying whether the agent is waiting for upload, in interactive mode, or processing/complete.

- Event handlers:
  - `on_file_upload(file)`:
    - Calls `agent.process_file(file)` and updates the chatbot and status box.
  - `on_msg_submit(user_msg, history)`:
    - Calls `agent.handle_chat(user_msg, history)` and updates the chat and status text accordingly.

- The app launches with public sharing enabled (`demo.launch(debug=True)` with autoâ€‘`share=True` on Colab), so it can be accessed via a public Gradio URL while the notebook runs.

## Running the Project

### Prerequisites

- Python environment (Colab recommended).
- Google account + Google Drive access.
- Google Generative AI API key (Gemini) with file upload capability.
- Dependencies:
  - `chromadb`
  - `sentence-transformers`
  - `pandas`, `numpy`, `scikit-learn`
  - `gradio`
  - `google-generativeai`

### Steps

1. **Run the web crawler (outside the notebook)**
   - Execute the crawler script that scrapes Simplify's GitHub job listings (intern + newâ€‘grad).
   - Ensure it outputs a CSV with the schema described above (e.g., `jobs_df_demo.csv`).
   - Place the CSV in your Google Drive under the expected path.

2. **Open and configure `JobsRecommendation.ipynb`**
   - Mount Google Drive in Colab.
   - Set `DB_PATH` and CSV path (`jobs_df_demo.csv`) if you change the defaults.
   - Run the initial cells to install packages, load the embedding model, and build the feature pipeline.

3. **Ingest jobs into ChromaDB**
   - Run the "Load VecDB" section.
   - On first run, this will:
     - Vectorize all jobs.
     - Create and populate the Chroma collection.
   - On subsequent runs, it will load the existing collection and re-instantiate the `FeatureVectorization` pipeline.

4. **Configure Gemini API key**
   - When prompted, enter your Google API key in the notebook input (or store it as a Colab secret named `GOOGLE_API_KEY`).

5. **Launch the Gradio app**
   - Run the "AI agent UI / Gradio" section.
   - Use the generated public Gradio URL:
     - Upload a PDF resume.
     - Answer follow up questions in chat.
