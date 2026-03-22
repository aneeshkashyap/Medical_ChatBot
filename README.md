# Medical Information Chatbot

## Overview

The Medical Information Chatbot is a dataset-driven healthcare assistant designed to provide general medical information using verified knowledge sources.
The system retrieves relevant medical information from a structured dataset and presents it in a clear prescription-style report for educational purposes.

The chatbot is intended to support users seeking reliable healthcare knowledge while clearly stating that the system does not perform medical diagnosis.

---

## Key Features

### Verified Healthcare Knowledge

Information is derived from structured medical datasets and supported with trusted references such as:

* WHO (World Health Organization)
* CDC (Centers for Disease Control and Prevention)
* NHS (National Health Service)
* AHA (American Heart Association)
* NIMH (National Institute of Mental Health)

---

### Intelligent Query Matching

User queries are processed using a similarity-ranking approach that evaluates:

* Keyword similarity
* Jaccard similarity
* Fuzzy text similarity
* Category relevance
* Intent detection

This allows the chatbot to retrieve the most relevant healthcare information even when the query is phrased differently.

---

### Symptom-Based Condition Inference

The system can interpret symptom descriptions provided by users and match them against symptom information in the dataset to infer possible conditions.

Example:
User input:
"My stomach burns after eating spicy food"

System inference:
Possible digestive condition with related guidance.

---

### Structured Health Report Output

Responses are presented in a structured format including:

* Patient name
* User query
* Probable condition
* Confidence score
* Description
* Causes
* Symptoms
* Care guidance
* Emergency warning signs
* Verified medical sources

---

### Research Transparency

The system includes several explainability features:

* similarity score breakdown
* top related condition matches
* evidence coverage indicators
* timestamped analysis reports

These features help demonstrate how the chatbot generated the response.

---

## System Architecture

User Query
↓
Text Normalization
↓
Keyword Extraction
↓
Intent Detection
↓
Similarity Ranking Algorithm
↓
Dataset Retrieval
↓
Medical Knowledge Integration
↓
Structured Health Report

---

## Technologies Used

* Python
* Pandas (dataset processing)
* Regular Expressions
* SequenceMatcher (fuzzy similarity)
* Text Processing

---

## Dataset

The chatbot uses a structured dataset containing healthcare questions and answers with category labels.

Example structure:

| Question           | Answer | Category    |
| ------------------ | ------ | ----------- |
| What is asthma     | ...    | respiratory |
| Symptoms of asthma | ...    | respiratory |
| What causes asthma | ...    | respiratory |

---

## Safety and Ethical Considerations

This system is designed for educational purposes only.

* The chatbot does **not diagnose diseases**
* It provides **general healthcare guidance**
* It recommends consulting medical professionals for serious symptoms

---

## Limitations

* Dataset-dependent knowledge
* No machine learning model
* Limited coverage of diseases
* Not intended for clinical decision making

---

## Future Improvements

* Integration with medical ontologies
* machine learning-based symptom classification
* expanded healthcare dataset
* multilingual healthcare support

---

## Author

Aneesh Kashyap K S
Sri Venkateshwara College of Engineering

---

## License

Educational research project.
