eISSN: 3063-802X ; pISSN: 3063-8011 **Vol. 3, No. 2, 2026 Hal. 01-10** 

**Available online at** _https://teewanjournal.com/index.php/juragan_ 

# **Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering** 

**Krisna Dwi Setya Adi[1*] , Ivan Michael Siregar[2 ]** Information System,  Institut Teknologi Harapan Bangsa, Bandung, Indonesia[1,2 ] 

* Email Correspondence: ds-19010@students.ithb.ac.id 

## **ARTICLE INFO** 

## **ABSTRACT** 

## **Article History:** 

_Submitted 10-07-2026 Received - Published -_ 

## _**Keywords:**_ 

_Retrieval-Augmented Generation; Question Answering; Multi-Source;_ FAISS; Knowledge Transfer; 

_Key personnel turnover creates knowledge gaps in document-based service organizations, where information is distributed across technical specifications, operational databases, and team discussions. This study develops a multi-source Retrieval-Augmented Generation (RAG)based Question Answering (QA) system that automatically integrates heterogeneous knowledge sources through a unified source parameter. Using the Adapter Pattern, the system converts PDF/TXT documents and PostgreSQL tables into a common representation, builds a FAISS vector index, retrieves relevant context, and generates grounded answers with Gemini 2.5 Flash. Evaluation employs eight metrics and three composite scores: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI). Experiments were conducted on the BOND_SYS dataset using 25 Indonesian questions covering specification documents, an 8-table PostgreSQL database, and 908 developer discussion messages. Results show perfect retrieval performance (Precision@K = 1.000; MRR = 1.000) across all scenarios. The full hybrid configuration achieves the highest Overall score (0.373), while Scenario C records the highest MSRS (0.825). Scenario E obtains ROUGE-L = 0.181 and BLEU-1 = 0.196 using five manually curated reference answers. Two baseline comparisons further support this contribution: a zero-shot LLM without retrieval correctly answered only 8% of questions, while a BM25 keyword-search baseline, competitive on single-source scenarios, was outperformed on cross-referencing tasks, underscoring the added value of dense multi-source retrieval. The findings demonstrate that integrating formal documents, structured databases, and discussion logs enhances knowledge transfer and question answering for organizational support and employee onboarding._ 

 1 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

## **INTRODUCTION** 

Organizations operating in document-based service ecosystems, such as financial institutions, domain-specific technical service providers, and transaction platform operators, rely heavily on knowledge accumulated across various artifacts: functional requirement specifications, operational procedures, system configurations, and transaction records. A critical challenge arises when key personnel turnover occurs: domain knowledge embedded in individuals must be transferred to successors within a limited time, while service operations cannot be interrupted. According to Chui et al. (2012), knowledge workers spend approximately 20% of their working time searching for internal information, and IDC (2023) reports that up to 90% of organizational data is unstructured scattered across PDF documents, text files, and databases, such that only a small fraction is effectively accessible and utilized. 

This problem is further compounded by the heterogeneous distribution of organizational knowledge: functional specifications are stored in PDF documents spanning tens to hundreds of pages, operational data resides in relational database tables, while the context of technical decision often exists only in the personal memory of the individual involved. Nonaka and Takeuchi (1995) distinguish between tacit knowledge (implicit in an individual’s mind). However, even explicit knowledge is difficult to locate quickly when it is scattered across repositories in different formats. Alavi and Leidner (2001) further emphasize that knowledge management systems often fail when organizational knowledge remains fragmented across repositories and cannot be operationalized in day-to-day decision workflows.  Gartner (2020) estimates that employees spend up to 30% of their working time on low-value activities that could be automated, including document retrieval and repetitive operational information clarification. 

In the enterprise environment that forms the context of this study, a financial instrument transaction platform manages the offering and allocation processes of financial instruments for registered participant institutions. The impact of turnover is felt directly at the operational level. Over a two-year period, two consecutive changes of Product Owner occurred, with the team size shrinking from three to two members. As a result, each technical clarification request from external users (platform participants), such as transaction process flows, configuration parameters, or instrument rules, required response times ranging from several hours to a full working day, depending on how thoroughly the context was documented and how familiar the remaining personnel were with the material. This condition represents a common pattern found across various document-based service organizations: the volume of knowledge does not decrease, but the human capacity to access it quickly and accurately becomes increasingly limited. 

Traditional knowledge management (KM) solutions such as internal wikis and static knowledge bases are unable to answer dynamic questions that require real-time cross-source inference. Users must still manually browse through tens to hundreds of pages of technical specifications and operational procedures while simultaneously checking configuration tables in the database; a process that consumes hours or even days before a conclusion can be formulated. Large Language Models (LLMs) without factual grounding present a different risk: Gao et al. (2024) identify three critical weaknesses of LLMs in domain-specific service context namely hallucination, outdated knowledge and non-transparent reasoning, while Ji et al (2023) provide broader evidence that hallucination remains a structural limitation in neutral text generation systems. Furthermore, the Self-RAG approach (Asai et al. 2023) explicitly trains LLMs to evaluate retrieval relevance and self-critique their own answer, confirming that standard LLMs without such mechanisms lend to produce answer without considering the boundaries of their knowledge, a behavior that is 

 2 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

particularly dangerous in service contexts demanding accuracy and traceability. A clear gap therefore exists: no solution is yet capable of answering factual questions accurately and promptly from heterogeneous sources already present within an organization, where “promptly” means users do not have to wait extended periods to reach a conclusion, without requiring different technical configurations for each source type. Two specific limitations define this gap: first, existing RAG systems are designed for a single source type and cannot transparently handle heterogeneous data; second conventional pre-indexed systems do not reflect real-time content changes without manual re-indexing. 

The Retrieval-Augmented Generation (RAG) approach introduced by Lewis et al. (2020) opens an opportunity to address these limitations by combining retrieval from external sources with LLM generation capabilities, enabling the system to provide factual, grounded answers based on actual documents. This foundation was subsequently strengthened by dense and generative passage retrieval methods (Karpukhin et al. 2020; Izacard and Grave 2021), knowledge-graph-augmented multi-hop reasoning (Yasunaga et al. 2021), scalable vector search infrastructure (Johnson et al. 2019; Douze et al. 2024), and multilingual semantic representation models (Reimers and Gurevych 2019). However, these classical RAG implementations are generally single-source and require different configurations for each data source type, thereby increasing the technical burden for organizations with heterogeneous data ecosystems.

Gao et al. (2024) and Sharma (2025) document a shift in the field toward handling this heterogeneity and adaptivity directly, rather than only improving retrieval within a single source type. Cheng et al. (2025) similarly survey knowledge-oriented RAG spanning unstructured text, semi-structured, and structured (graph) knowledge. At the architectural level, two concurrent systems illustrate this shift: HetaRAG (Yan et al. 2025) orchestrates retrieval across heterogeneous data stores by unifying vector indices, knowledge graphs, full-text engines, and structured databases, while HyPA-RAG (Kalra et al. 2025) introduces query-complexity-aware adaptive parameter selection to balance retrieval cost and accuracy in high-stakes document QA. These systems confirm that multi-source and heterogeneous retrieval is an active research frontier. However, they generally rely on multiple specialized stores, such as a vector database, a graph database, and a full-text index, maintained in parallel, which increases infrastructure and maintenance complexity. This study addresses that same heterogeneity through a single Adapter Pattern interface instead of multiple parallel stores. Evaluating RAG-based QA systems requires a dual perspective: retrieval quality using Precision@K and MRR (Voorhees 1999), generation quality using ROUGE-L (Lin 2004) and BLEU-1 (Papineni et al. 2002), and RAG-specific dimensions proposed by Es et al. (2023), namely faithfulness and answer relevance. 

These limitations motivate the present study. This study develops a RAG-based QA system that is agnostic to data sources, capable of handling unstructured documents (PDF, TXT) and relational databases (PostgreSQL) in a unified manner through a single configuration interface (Adapter Pattern), thereby providing a knowledge transfer infrastructure that can be directly implemented in document-based service organizations. The contributions of this study are summarized as follows: 

1. Propose an integrated source-agnostic RAG architecture model that is layered-based and  works under various source configurations and  real-time conditions. 

2. Implement a layered-based of structural and unstructured knowledge paradigm with progressive scenarios that isolate and quantify the incremental contribution of each layer. 

3. Evaluate the model using metrics composite evaluation metrics purpose-build for multi-source RAG in organizational knowledge transfer contexts: Knowledge Transfer Effectiveness (KTE), Multi-Source Retrieval Score (MSRS), and Answer Quality Index (AQI) 

## **RESEARCH METHODS** 

## **Data Sources and Preprocessing** 

This study employs real operational data from a domain-specific financial instrument transaction platform (BOND_SYS) organized in a three-layer corpus: (L1) BOND_SYS system specifications in PDF/TXT document format, covering system module descriptions, business process flows, and technical requirements; (L2) a BOND_SYS PostgreSQL database with 8 operational tables containing real data (20 

 3 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

RFQs, 10 Securities, 10 firms, 10 quotations, 10 trades, 10 trade statuses, 11 firm default parameters, 8 fraction masters); and (L3) developer team discussion logs in TXT format comprising 908 messages from three sources: a 2022 group discussion, a February 2025 group discussion and a 2022 personal conversation. System, institutional and individual identities are anonymized using a token masking scheme (system name: BOND_SYS, ministry name: GOV_DEPT1, module name: BOND_MOD, etc). 

Preprocessing is performed by two adapters according to source type. FolderSourceAdapter handles document files using the appropriate library per format (pypdf for PDF, built-in for text). All extracted text is normalized into RawDocument objects with content, source and format metadata attributes. 

PostgreSQLAdapter handles relational sources in three modes: (1) all tables, (2) specific tables via a table name list (pg_tables), and (3) custom SQL queries (pg_queries). Each table or query result is converted into structured text that includes column names, row counts, and tabular data, enabling embedding and retrieval by FAISS. In this study, mode (2) is used with a predefined list of 8 BOND_SYS operational tables; modes (1) and (3) are supported by the adapter interface but were not invoked in this evaluation, as the relevant tables were known in advance and no cross-table aggregation queries were required. 

Following extraction, text is segmented using UniversalTextSplitter based on RecursiveCharacterTextSplitter with chunk_size = 2000 and overlap = 300. The 300 character overlap is designed to preserve inter-chunk context so that information split at chunk boundaries is not entirely lost. The larger chunk_size (2000) is selected to support Indonesian-language financial documents that typically contain long sentences and multi-row tables. This chucking choice is aligned with retrieval-augmented pretraining evidence that context segmentation and retrieval granularity substantially affect downstream generation quality and factual grounding (Borgeaud et al. 2022; Shi et al. 2023) Table 1 summarises the file formats supported by FolderSourceAdapter. 

**Table 1. File formats supported by FolderSourceAdapter** 

|**_Format_**|**_Library_**|**_Processing_**|
|---|---|---|
|_.pdf_|pypdf|Text extraction<br>from all pages|
|_.txt, .md, .log_|built-in|Raw text|



The selection of retrieval hyperparameters is justified as follows. The top-K parameter is set to 8, balancing between recall (more chunks = more potential evidence) and precision (fewer chunks = less noise in the LLM context window). Empirical testing showed that K < 5 frequently missed relevant chunks in multi-source scenarios, while K > 10 introduced irrelevant context that reduced Answer Faithfulness. The similarity threshold of 0.25 (cosine distance) was calibrated through iterative testing on development queries: thresholds below 0.20 admitted chunks with minimal semantic relevance, while thresholds above 0.35 excluded moderately relevant chunks critical for Answer Completeness. The chunk_size of 2000 characters accommodates the structure of Indonesian-language financial documents, which typically 

 4 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

contain compound sentences averaging 40-60 words and multi-row configuration tables that would be fragmented at smaller chunk sizes (e.g., 500 or 1000 characters). The overlap of 300 characters (15% of chunk_size) ensures that sentences spanning chunk boundaries maintain sufficient context for coherent retrieval. 

## **Evaluation Dataset** 

The evaluation covers five scenarios, each representing a different source type and knowledge transfer dimension. All prompts and system responses are derived from real operational data in Indonesian, anonymized from the BOND_SYS platform. To preserve linguistic fidelity, prompts shown in the figures and tables are presented in their original Indonesian, with English translations provided in italics for accessibility. Result excerpts, however, are presented in English to maintain consistency with the analytical discussion throughout the paper. 

Scenario A, Chat Only (L3, tacit to operational): Developer team discussion logs from BOND_SYS (908 messages, 3 TXT files). Five questions cover operational issues found exclusively in the discussion logs: submit quotation bug, upload allocation demo issue, a filter-status bug tracked as NEWCORE-2442 (an internal issue tracker ticket referencing a backend detect in the WAITING-status filter), decimal digit decision, and amend feature status. 

Scenario B, PDF Only (L1, explicit to actionable): BOND_SYS system specification documents in PDF/TXT format. Five questions cover documented business process flows: INSTRUMENT_TYPE_A stages, differences between General/Restricted sessions, parties involved in RFQ approval, Upload Allocation technical requirements, and broadcast notification mechanism. 

Scenario C, PostgreSQL Only (L2, explicit to structured): BOND_SYS PostgreSQL database with 8 operational tables. Five questions cover structural configuration data: default price percentage values, fraction_type/digit combinations, auction_unit differences per board, firms with is_active = Y, and quotations allocated to a specific RFQ. 

Scenario D, PDF + DB (L1+L2, explicit to cross-referenced): Combined PDF documents and PostgreSQL tables (8 tables) through MultiSourceAdapter. Five questions require cross-referencing specification documents with actual data: session time consistency, board type in documents vs. DB, offering_parameter consistency, settlement_date calculation, and offering_digit consistency with fraction_masters. 

Scenario E, Hybrid All (L1+L2+L3, Cross-Paradigm): All layers (PDF, PostgreSQL, and discussion logs) are combined into a single FAISS index. Five questions require all three layers simultaneously: submit quotation BOARD_TYPE_A bug (Chat+PDF), offering digit decision (Chat+DB), BOND_SYS ETL incident (anonymized period) (Chat), upload allocation flow (PDF+DB+Chat), and amend feature status (PDF+DB+Chat). This is the only scenario equipped with 5 manually curated reference answers (GROUND_TRUTH_HYBRID), making ROUGE-L and BLEU-1 reference-based, in contrast to Scenarios A-D which are reference-free (vs. retrieved context). 

## **System Architecture** 

The system is built using the Adapter Pattern (Gemma E et al., 1994) to enable data source 

 5 

**eISSN 3063-802X & pISSN 3063-8011** _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and**_ e%®o ~~SS~~ _**Database Question Answering**_ **(Adi, et al.)** Ge ~~JURACAN ea OO ———————————EEEEEAA PEN CABDIAN - PENELITIAN~~ 

extensibility without modifying the core pipeline. The architecture consists of seven main components arranged sequentially: SourceDetector detects the source type from the input parameter, SourceFactory initializes the appropriate adapter (FolderSourceAdapter or PostgreSQLAdapter), UniversalTextSplitter segments the extracted text, RuntimeIndexBuilder constructs the FAISS index in-memory, QueryProcessor performs similarity search on the index, and AnswerGenerator produces answers along with evaluation of 8 metrics simultaneously. The overall architecture is illustrated  in Figure 1. 

**Figure 1.** Architecture of the Agnostic Multi-Source RAG System 

Inputs (user question, source parameter, Config) enter the pipeline through SourceDetector, which classifies the source string and delegates document loading to the corresponding adapter via SourceFactory: FolderSourceAdapter for local files (PDF/TXT/MD/LOG), PostgreSQLAdapter for relational tables (SELECT * LIMIT 1000, serialized to text), or MultiSourceAdapter for hybrid sources. Raw documents are split by UniversalTextSplitter into 2,000-character chunks (300 character overlap), encoded by a singleton multilingual MiniLM-L12-V2 model (384 dimensions), and indexed in a FAISS in memory vector store with session-level caching. At query time, QueryProcessor embeds the question, retrieves top8 chunks (similarity threshold 0.2), and passes the formatted context to AnswerGenerator, which calls 

 5 6 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

Gemini 2.5-flash with zero-shot prompting and an automatic retry-and-model-fallback chain (attempt 2 falls back to gemini-2.0-flash, attempt 4 to gemini 1.5 flash; exponential backoff with base delay = 15 s, max 6 entries. The final RAG Results is simultaneously evaluated on 8 quantitative metrics: Retrieval Relevance (RR), Answer Faithfulness (AF), Answer Completeness (AC), ROUGE-L, BLEU-1, Precision@K, MRR, and Context Coverage (CC). Directed edges represent data flow; dashed edges indicate parallel scoring path in which intermediate  retrieval artifacts (chunks and relevance scores) are forwarded to the Evaluator independently of the generation step. 

SourceDetector applies pattern matching rules to automatically detect the source type, as summarised in Table 2. 

**Table 2** . **Automatic detection rules for data source type** 

|**_Input Pattern_**|**_Adapter_**|**_Example_**|
|---|---|---|
|postgresql:// or postgres://|PostgreSQLAdapter|postgresql://user:pass@host/db|
|Absolute path(/, C:\, ~)|FolderSourceAdapter|/content/drive/Mydrive/data|
|Path relatif (./, ../)|FoldersourceAdapter|./documents/laporan|
|Default fallback|FolderSourceAdapter|Nama folder tanpa prefix|



The embedding component uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, CPU, normalize_embeddings=True) as a singleton loaded once and reused throughout the pipeline. This model supports over 50 languages including Indonesian. The generative component uses Gemini 2.5-flash as the primary model with google/flan-t5-base as a fallback. 

## **Model Training and Evaluation** 

The RAG system does not undergo training (fine-tuning) phase, as it employs pre-trained models covering multilingual domains. The research focus is on end-to-end pipeline performance evaluation using eight quantitative metrics. 

The FAISS index is constructed in-memory (real-time indexing) at each invocation of pipeline.ask(), unlike conventional RAG systems that persist the index to disk. The execution sequence is: (1) the adapter loads documents from the source on-the-fly, (2) the splitter segments the text, (3) FAISS builds the index in RAM, (4) the query processor performs retrieval, and (5) the answer generator produces the answer. Session cache (use_session_cache=True) enables index reuse for the same source within a single execution session, reducing overhead without sacrificing inter-call data consistency within the same session. This design choice follows the practical FAISS implementation direction described in recent library documentation (Douze et al. 2024). 

## **Interpretive Analysis and Evaluation Design** 

 7 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

The 8-metric evaluation framework is grouped into three dimensions to enable separate interpretation of retrieval quality and generation quality: 

**Retrieval Dimension (Classical IR):** Precision@K Eq. (1) measures the proportion of top-K retrieved chunks that are relevant. MRR Eq. (2) captures rank quality by rewarding systems that place the first relevant chunk higher. Context Coverage Eq. (3) measures source diversity across top-K chunks. 

**==> picture [318 x 25] intentionally omitted <==**

**==> picture [329 x 26] intentionally omitted <==**

**==> picture [325 x 24] intentionally omitted <==**

## **Answer Quality Dimension:** 

- Retrieval Relevance = cosine similarity of query embedding vs. mean of top-K chunk embeddings 

- Answer Faithfulness = F1 token overlap of answer vs. combined context (anti-hallucination) 

- Answer Completeness = ratio of question keywords present in the answer 

## **NLP Dimension:** 

- ROUGE-L (Lin 2004) = F1 based on Longest Common Subsequence 

- BLEU-1 (Papineni et al. 2002) = Unigram precision with brevity penalty 

In addition to these eight metrics, three composite metrics are defined to address evaluation questions that cannot be answered by any single metric: 

Composite Metric 1, Knowledge Transfer Effectiveness (KTE) 

**==> picture [218 x 36] intentionally omitted <==**

**==> picture [281 x 17] intentionally omitted <==**

KTE is the average of two components Eq. (4): Faithfulness (proportion of the answer supported by context, as an anti-hallucination measure) and Completeness (proportion of question keywords covered in the answer). The effective threshold is set at KTE  0.5. Knowledge transfer succeeds only if the answer is simultaneously non-hallucinatory and complete. If either component is zero, knowledge transfer fails; the simple average is therefore sufficient as an effectiveness measure (Nonaka and Takeuchi 1995). 

Composite Metric 2, Multi-Source Retrieval Score (MSRS): 

**==> picture [347 x 36] intentionally omitted <==**

MSRS is the average of two components Eq. (5): Precision@K (proportion of relevant chunks in top-K retrieval) and Context Coverage (source document diversity in top-K). A system that retrieves only 

 8 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

from a single file will obtain a low Context Coverage even with a high Precision@K. MSRS detects this condition and ensures that the multi-source claim is evidenced at the retrieval level. This approach applies the diversity principle in Information Retrieval (Carbonell and Goldstein 1998) to the multi-source RAG context. 

Composite Metric 3, Answer Quality Index (AQI): 

**==> picture [365 x 36] intentionally omitted <==**

AQI is the average of three components Eq. (6): Faithfulness (anti-hallucination), Completeness (question coverage), and ROUGE-L (structural similarity based on word sequence order against context). While KTE only measures whether knowledge is conveyed in terms of content, AQI adds a linguistic dimension to detect answers that are thematically adequate but structurally divergent from the source documents. This multi-aspect approach is consistent with the SummEval methodology (Fabbri et al. 2021). 

The relationships among composite metrics: KTE measures from the user perspective (whether knowledge is conveyed), MSRS from the system perspective (whether multi-source is evidenced), and AQI from the linguistic perspective (whether the answer is of NLP quality). The three are complementary; a well-performing system should achieve high scores across all three dimensions simultaneously. 

The aggregate Overall metric is calculated as the simple average of five answer quality metrics: 

**==> picture [405 x 36] intentionally omitted <==**

Pure retrieval metrics (Precision@K, MRR, Context Coverage) are excluded from Overall Eq. (7) so that this aggregate reflects answer quality rather than retrieval quality, which is consistently perfect (P@K = MRR = 1.000). 

This three-dimensional evaluation separation enables identification of whether low values are caused by retrieval failure or by limitations in the generative model’s language capability, two problems that require fundamentally different solutions. 

To substantiate the “Multi-Source” claim, the evaluation is designed in five scenarios, each representing one dimension of organizational knowledge transfer. Scenario E specifically proves the source-agnostic claim at the highest level: a single query spans PDF/TXT system specification documents, team discussion logs (TXT), and PostgreSQL tables (SQL) simultaneously within a single FAISS index. Scenario E is also the only scenario equipped with ground truth, making ROUGE-L and BLUE-1 referencebased. Table 3 summarises the five scenarios with their respective adapters and knowledge transfer dimensions 

**Table 3. Multi-Source evaluation design and knowledge transfer dimensions** 

|**_Scenario_**|**_Layer_**|**_Adapter_**|**_Source_**|**_Format_**|**_KT_**|**_Knowledge_**|
|---|---|---|---|---|---|---|
||||||**_Dimension_**|**_Type_**|



 9 

**eISSN 3063-802X & pISSN 3063-8011** 

## _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|A: Chat|L3|_FolderSourceAdapter_|BOND_SYS<br>discussion log<br>(908<br>messages, 3<br>TXT file )|TXT|Tacit to<br>Operational|Percakapan<br>informal⇒<br>jawaban<br>operasional|
|---|---|---|---|---|---|---|
|B: PDF|L1|_FolderSourceAdapter_|BOND_SYS<br>system spec (<br>PDF docs)|PDF./TXT Explicit to|PDF./TXT Explicit to<br>Actionable|Format<br>specification to<br>business process<br>insight|
|C:<br>PostgreSQL|L2|_PostgreSQLAdapter_|BOND_SYS<br>DB  (8 tables,<br>20 RFQ)|SQL|Explicit to<br>Structured|Table data to<br>contextual<br>narrative|
|D: PDF +<br>DB|L1 +<br>L2|_MultiSourceAdapter_|PDF +<br>PostgreSQL<br>(combined)|TXT  +<br>SQL|Explicit to<br>Cross-<br>referenced|Cross verification<br>of specification<br>and actual data|
|E: Hybrid|L1 +<br>L2 +<br>L3|_MultiSourceAdapter_|PDF + DB +<br>Chat (all<br>layers)|TXT +<br>SQL|Cross-<br>Paradigm|Tacit, explicit,<br>and structured<br>combined|



Scenario E: reference-based evaluation (ROUGE-L and BLEU-1 vs. GROUND_TRUTH_HYBRID). Scenarios A-D: reference-free (vs. retrieved context). 

## **RESULT AND DISCUSSION** 

## **System Performance** 

## **1. Multi Source** 

An initial evaluation was performed on the question “What did the team discuss regarding the NEWCORE-2442 issue and what is the resolution status based on the conversation log?” using the BOND_SYS developer team discussion logs (908 messages, 3 TXT files) as the source, with Gemini 2.5flash (Google, multilingual) as the generative model. This question represents a realistic knowledge retrieval scenario in a software development environment, where users need to identify discussion related to a specific issue and determine its resolution status without manually reviewing hundreds of messages. Evaluation results are reported in Table 4. 

**Table 4. Single-query evaluation results (question A3) using Gemini 2.5-flash** 

|**_Metric_**|**_Gemini 2.5 Flash_**|**_Gemini 2.5 Flash_**|**_Gemini 2.5 Flash_**|**_Dimension_**|
|---|---|---|---|---|
|Retrieval Relevance|0.460|||Retrieval|



 10 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|Answer Faithfulness|0.064|Generation|
|---|---|---|
|Answer Completeness|0.833|Generation|
|ROUGE-L|0.029|Generation|
|BLEU-1|0.000|Generation|
|Precision@K|1.000|Retrieval|
|MRR|1.000|Retrieval|
|Overall|0.277|Combined|



The retrieval component achieved strong ranking performance, as indicated by perfect Precision@K(1.000) and MRR (1.000) scores, demonstrating that the relevant context was successfully retrieved and ranked at the top position. However, the Retrieval Relevance score of 0.460 suggests that the retrieved documents were only moderately aligned with the query semantics. 

On the generation side, the system achieved high Answer Completeness (0.833), indicating that the generated response covered most of the information required to answer the question. Nevertheless, the low Answer Faithfulness score (0.064), together with low ROUGE-L (0.029) and BLEU-1 (0.000) values, suggests that the response was highly abstractive and did not closely match the wording of the source documents or reference answer. This behaviour is expected when using large language models to summarize conversational data, particularly in multilingual and informal communication contexts where semantically correct answers may differ substantially from the original phrasing. 

## **2. Multi-Source Batch Evaluation** 

To substantiate the Multi-Source claim and measure knowledge transfer effectiveness, a batch evaluation was conducted across five scenarios with 25 questions in total (5 per scenario). Scenarios A, B, and C evaluate each corpus layer separately. Scenario D evaluates the L1 + L2 combination and Scenario E evaluates all three layers simultaneously through MultiSourceAdapter. Scenario E is the only one that employs reference-based evaluation (against 5 manually curated reference answers), while Scenarios A-D use reference-free evaluation (against retrieved context). All results are produced below using the Gemini 2.5-flash model. 

**Table 5a. Retrieval summary per scenario** 

|**_Scenario_**|**_Adapter_**|**_Format_**|**_N_**|**_RR_**|**_P@K_**|**_MRR_**|**_CC_**|
|---|---|---|---|---|---|---|---|
|A: Chat<br>Only (L3)|_FolderSourceAdapter_|TXT|5|0.473|1.000|1.000|0.450|



 11 

## **eISSN 3063-802X & pISSN 3063-8011** _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|B: PDF (L1)|_FolderSourceAdapter_|TXT|5|0.580|1.000|1.000|0.250|
|---|---|---|---|---|---|---|---|
|C:<br>PostgresQL<br>(L2)|_PostgresQLAdapter_|SQL|5|0.460|1.000|1.000|0.650|
|D: PDF +<br>DB (L1+L2)|_MultiSourceAdapter_|TXT + SQL|5|0.576|1.000|1.000|0.525|
|E: Hybrid<br>(L1+L2+L3)|_MultiSourceAdapter_|TXT + SQL|5|0.582|1.000|1.000|0.425|



_RR = Retrieval Relevance; CC = Context Coverage; P@K dan MRR = 1.000 across all scenarios._ 

**Table 5b. Answer quality and composite metrics per scenario** 

|**_Scenario_**|**_Faith_**|**_Comp_**|**_ROUGE-_**<br>**_L_**|**_BLEU-_**<br>**_1_**|**_Overall_**|**_KTE_**|**_MSRS_**|**_AQI_**|
|---|---|---|---|---|---|---|---|---|
|A: Chat (L3)|0.075|0.853|0.026|0.000|0.285|0.464|0.725|0.318|
|B: PDF (L1)|0.114|0.675|0.046|0.000|0.284|0.395|0.625|0.279|
|C:<br>PostgreSQL<br>(L2)|0.0.44|0.581|0.022|0.000|0.221|0.312|0.825|0.216|
|D: PDF+DB<br>(L1+L2)|0.117|0.661|0.041|0.000|0.279|0.389|0.762|0.273|
|E: Hybrid<br>(L1+L2+L3)|0.127|0.779|0.181|0.196|0.373|0.453|0.713|0.362|



F _aith = Answer Faithfulness; Comp = Answer Completeness. KTE = (Faith + Comp) / 2. MSRS = (P@K + CC ) / 2. AQI = (Faith + Comp + ROUGE-L) / 3. Overall = (RR + Faith + Comp + ROUGE-L + BLEU-1) / 5 Skenario E: ROUGE-L dan BLEU-1 reference-based. Scenarios A-D: reference-free. All values rounded to three decimal places; rounding differences of 0.001 may occur in composite formulas._ 

To substantiate the Multi-Source claim and measure knowledge transfer effectiveness, a batch evaluation was conducted across five scenarios with 25 questions in total (5 per scenario). Scenarios A, B, and C evaluate each corpus layer separately. Scenario D evaluates the L1 + L2 combination and Scenario E evaluates all three layers simultaneously through MultiSourceAdapter. Scenario E is the only one that 

 12 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

employs reference-based evaluation (against 5 manually curated reference answers), while Scenarios A-D use reference-free evaluation (against retrieved context). All results are produced below using the Gemini 2.5-flash model. 

## **3. Ablation Study: Layer Contribution Analysis** 

To demonstrate that each corpus layer provides a measurable contribution, an ablation study was conducted on the same questions (E1-E5) across four progressively expanded source configurations. 

**Table 6. Ablation study results (per-layer contribution)** 

|**_Configuration_**|**_Active_**<br>**_Layer_**|**_n_**|**_Overall_**|**_Faithfulness_**|**_Completeness_**|**_ROUGE-_**<br>**_L_**|**_BLEU-_**<br>**_1_**|
|---|---|---|---|---|---|---|---|
|Ablation-0:<br>Chat|L3|5|0.237|0.076|0.636|0.026|0.00|
|Ablation-1:<br>PDF|L1|5|0.214|0.139|0.320|0.053|0.000|
|Ablation-2:<br>PDF + DB|L1+L2|5|0.234|0.128|0.428|0.051|0.002|
|Full: PDF +<br>DB + Chat|L1+L2+L3|5|0.373|0.127|0.779|0.181|0.196|



Ablation-0 through Ablation-2: reference-free (vs. retrieved context). Full (L1+L2+L3): reference-based (vs. GROUND_TRUTH_HYBRID). Overall = (RR + Faithfulness + Completeness + ROUGE-L + BLEU-1) / 5; the RR column is not displayed as it is collected alongside other metrics. Ablation-Full is identical to Scenario E in Tables 5a/5b (RR = 0.582). 

The ablation results show a clear contribution from combining all three knowledge layers. The Overall score remains relatively similar across Chat-Only (0.237), PDF-only (0.214), and PDF+db (0.234), but increases substantially to 0.373 when the complete three-layer configuration (L1+L2+L3) is used, representing a gain of +0.139 over PDF+DB. Similar improvements are observed in ROUGE-L, which increases from 0.051 to 0.181 and BLEU-1, which rises from 0.002 to 0.196 after Layer 3 (Chat) is integrated. 

These findings indicate that conversational knowledge contained in team discussion logs provides information that is not fully represented in formal documentation or structured databases. The improvement is particularly evident for operational and historical questions requiring contextual knowledge, including quotation issues, offering digit decisions, ETL incident investigations, allocation workflows and feature implementation status. Therefore, the result suggests that the effectiveness of the proposed system is derived not only from the multi-adapter architecture but also from the integration of complementary knowledge layers within the corpus. 

## **Baseline Comparison: RAG vs. Zero-shot LLM** 

To provide an additional empirical comparison point beyond the qualitative comparison in Table 11, a zero-shot LLM baseline (Gemini 2.5-flash answering the same 25 questions without any retrieval step) was evaluated using the identical 8-metric framework. This closed-book-versus-retrieval-augmented comparison design follows the precedent set by Lewis et al. (2020), who evaluate RAG against a parametric-knowledge-only baseline to isolate the contribution of retrieval itself. Because this baseline retrieves no chunks, Retrieval Relevance, Precision@K, MRR, Context Coverage, and Answer Faithfulness are structurally zero by construction (they are defined as functions of retrieved chunks) rather than measured outcomes; the informative comparison points are Answer Completeness across all 25 questions and, for the 5 ground-truth Hybrid questions (E1-E5), ROUGE-L and BLEU-1 against `GROUND_TRUTH_HYBRID`. 

**Table 7. Baseline (zero-shot, no retrieval) vs. best multi-source scenario** 

|**_Metric_**|**_Baseline: Zero-shot (No RAG)_**|**_Scenario E: Hybrid (L1+L2+L3)_**|
|---|---|---|
|Answer Completeness (n=25)|0.134|0.779|
|ROUGE-L (5 ground-truth questions)|0.048|0.181|
|BLEU-1 (5 ground-truth questions)|0.007|0.196|
|Overall (n=25)|0.029|0.373|
|KTE (n=25)|0.067|0.453|

The zero-shot baseline scored far below Scenario E on every comparable dimension. Most strikingly, 23 of the 25 questions (92%), including all 5 ground-truth Hybrid questions, elicited the model's built-in refusal response ("Information not found in the available data sources") rather than a fabricated answer; the only two questions the model attempted to answer from parametric knowledge alone were generic, domain-independent business questions (RFQ approval roles, broadcast notification mechanisms), not questions tied to BOND_SYS-specific facts, identifiers, or team decisions. This confirms two points relevant to the study's contribution: (1) the organization-specific knowledge captured through the FolderSourceAdapter, PostgreSQLAdapter, and chat logs is not recoverable from the LLM's pretrained knowledge alone, so retrieval grounding is doing genuine work rather than being redundant with what a modern LLM already "knows"; and (2) the low Answer Faithfulness/refusal behaviour is a desirable property under this study's prompt design (Section "System Architecture") rather than a failure, since it shows the model prefers declining to answer over hallucinating specific organizational facts when no context is supplied, consistent with the conservative behaviour already noted in Pattern 1 of the Error Analysis. 

## **Comparison with Traditional Keyword Search (BM25)** 

While Table 7 establishes that retrieval grounding is necessary relative to no retrieval at all, it does not address whether the dense, semantic retrieval mechanism (FAISS) used by this system is itself necessary relative to a traditional, non-semantic retrieval mechanism. To test this directly, a second baseline replaces FAISS with BM25 (Robertson and Zaragoza 2009), a classical term-frequency keyword-matching retriever, while keeping the generator (Gemini 2.5-flash), the prompt template, and the `Evaluator` identical to Scenario A-E; only the retrieval mechanism differs. This dense-versus-sparse retrieval comparison design follows the precedent set by Karpukhin et al. (2020), whose Dense Passage Retrieval (DPR) study evaluates dense retrieval directly against BM25 to isolate the contribution of the retrieval mechanism itself. This baseline was run across the same five scenarios and 25 questions used throughout the study, directly testing the claim already made in the Introduction that "traditional knowledge management (KM) solutions such as internal wikis and static knowledge bases are unable to answer dynamic questions that require real-time cross-source inference." Because BM25 scores are not on the same scale as FAISS cosine similarity, Precision@K and MRR for this baseline use a BM25-native definition (a chunk counts as a match if its BM25 score is greater than zero, that is, it shares at least one keyword with the query) rather than the cosine-similarity threshold used elsewhere; the primary comparison is instead based on answer-quality metrics, which depend only on the retrieved context and the generated answer and are therefore retriever-agnostic. 

**Table 8. BM25 keyword-search retrieval vs. FAISS semantic retrieval, per scenario** 

|**_Scenario_**|**_Retriever_**|**_Answer Completeness_**|**_ROUGE-L_**|**_BLEU-1_**|**_Overall_**|**_KTE_**|**_Retrieval Relevance_**|
|---|---|---|---|---|---|---|---|
|A: Chat (L3)|FAISS (this study)|0.853|0.026|0.000|0.285|0.464|0.473|
|A: Chat (L3)|BM25|0.880|0.041|0.000|0.283|0.492|0.391|
|B: PDF (L1)|FAISS (this study)|0.675|0.046|0.000|0.284|0.395|0.580|
|B: PDF (L1)|BM25|0.833|0.093|0.001|0.326|0.530|0.474|
|C: PostgreSQL (L2)|FAISS (this study)|0.581|0.022|0.000|0.221|0.312|0.460|
|C: PostgreSQL (L2)|BM25|0.642|0.025|0.000|0.225|0.355|0.392|
|D: PDF+DB (L1+L2)|FAISS (this study)|0.661|0.041|0.000|0.279|0.389|0.576|
|D: PDF+DB (L1+L2)|BM25|0.658|0.031|0.000|0.239|0.375|0.415|
|E: Hybrid (L1+L2+L3)|FAISS (this study)|0.779|0.181|0.196|0.373|0.453|0.582|
|E: Hybrid (L1+L2+L3)|BM25|0.708|0.284|0.254|0.366|0.410|0.472|

The results are mixed rather than a uniform win for either retriever, and this pattern is itself informative. On Retrieval Relevance, the cosine-similarity-based measure of semantic alignment between the query and the retrieved chunks, FAISS is higher than BM25 in every scenario (e.g., 0.582 vs. 0.472 for Scenario E), which is expected since FAISS explicitly optimizes for this measure while BM25 optimizes for lexical overlap instead. However, this consistent semantic-relevance advantage does not translate into a consistent advantage in downstream answer quality: BM25 achieves a higher Overall score in Scenario B (0.326 vs. 0.284) and a higher ROUGE-L and BLEU-1 in Scenario E (0.284 vs. 0.181 and 0.254 vs. 0.196, respectively), while FAISS's clearest and largest advantage appears in Scenario D, PDF+DB cross-referencing (Overall 0.279 vs. 0.239), the scenario that most directly exercises the multi-source, cross-referencing capability this study is designed around. A plausible explanation is that the BOND_SYS corpus is dense with exact technical terminology (column names, board codes, module identifiers) that a query frequently repeats verbatim, a condition under which classical term-matching retrieval is known to remain competitive with dense retrieval (Robertson and Zaragoza 2009); the advantage of semantic retrieval becomes most visible precisely when the required evidence is dispersed across structurally dissimilar sources and cannot be located by keyword overlap alone, as in Scenario D. This adds nuance to, rather than overturns, the Introduction's claim about traditional KM and keyword-based search: BM25 is a substantially more capable baseline than the complete absence of retrieval evaluated in Table 7, but it is not adequate on its own for the cross-referencing task that motivates this study's multi-source architecture — and it is precisely on that task that dense retrieval demonstrates its clearest empirical value. 

 13 

**eISSN 3063-802X & pISSN 3063-8011** _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** GsJURNAL RAGAMJURACANPENGABDIAN - PENELITIAN 

## **4. Running Interface example and Prompt-Level Results** 

To complement aggregate quantitative metrics, this subsection presents one running-interface example and a prompt-level summary table. The running interface screenshot is taken from Scenario A (Chat Only), showing one complete interaction consisting of a user prompt and system response in the original operational language. 

**Figure 2.** Running-interface example (Scenario A): prompt and response are shown in original Indonesian. 

Table 9 summarises five prompt-level examples from Scenario A (Chat Only). Prompt and result texts are presented in Indonesian (original), while performance metrics are reported in English. 

**Table 9. Prompt-level results (scenario A: chat only)** 

|**_No._**|**_Prompting_**|**_Result_**|**_Performance_**|
|---|---|---|---|
|1|Masalah submit<br>quotation board<br>BOARD_TYPE_A<br>menjelang demo dan<br>workaround sementara ?<br>_Problem submitting_<br>_quotation board_<br>_BOARD TYPE A before_<br>_demo and temporary_<br>_workaround?_|The answer did not find sufficient contextual<br>evidence to explain the details of the problem and<br>the workaround.|RR=0.441;<br>P@K=1.000;<br>MRR=1.000;<br>Overall=0.269;<br>KTE=0.449|
|2|Mengapa upload<br>allocation tidak dapat|The answer identifies, based on the team's<br>discussion log on July 27, 2022, that the upload|RR=0.516;<br>P@K=1.000;|



 ry 14 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|didemonstrasikan pada|allocation feature could not be demonstrated|MRR=1.000;|
|---|---|---|
|sesi demo 27 Juli 2022|because of a feature update from DEV_E, which|Overall=0.282;|
||rendered the file unusable for upload allocation.|KTE=0.437|
|_Why couldn't upload_|||
|_allocation be_|||
|_demonstrated during the_|||
|_demo session on July 27,_|||
|_2022?_|||



||didemonstrasikan pada<br>sesi demo 27 Juli 2022<br>_Why couldn't upload_<br>_allocation be_<br>_demonstrated during the_<br>_demo session on July 27,_<br>_2022?_|allocation feature could not be demonstrated<br>because of a feature update from DEV_E, which<br>rendered the file unusable for upload allocation.|MRR=1.000;<br>Overall=0.282;<br>KTE=0.437|
|---|---|---|---|
|3|Apa yang didiskusikan<br>tim terkait isu<br>NEWCORE-2442 dan<br>apa status<br>penyelesaiannya<br>berdasarkan log<br>percakapan?<br>_What did the team_<br>_discuss regarding the_<br>_NEWCORE-2442 issue,_<br>_and what is its_<br>_resolution status based_<br>_on the conversation log?_|The answer captures the following.<br>The team discussed the NEWCORE-2442 issue<br>regarding:<br>•<br>A backend gap in filtering the WAITING<br>status and outputting the CLOSED and<br>ALLOCATED statuses.<br>•<br>A question on whether the feature needs to<br>operate in real time.<br>Based on the conversation log, the resolution<br>status is as follows.<br>Tim mendiskusikan isu NEWCORE-2442 terkait<br>dengan:<br>•<br>The backend still requires support for<br>filtering the WAITING, CLOSED, and<br>ALLOCATED statuses.<br>•<br>The feature is not yet real-time.<br>•<br>There was a discussion on adding a<br>WAITING status to the database and the<br>backend requirements needed for it.|RR=0.460;<br>P@K=1.000;<br>MRR=1.000;<br>Overall=0.277;<br>KTE=0.449|
|4|Apa keputusan teknis<br>yang didiskusikan tim<br>terkait jumlah desimal<br>(digit), untuk last price<br>dan offering price ?<br>_What technical decision_<br>_did the team discuss_<br>_regarding the number of_<br>_decimal digits for last_<br>_price and offering_<br>_price?_|The answer identifies the following,<br>Based on the given context, the technical decision<br>discussed by the team regarding the number of<br>decimal digits for last price and offering price is<br>as follows:<br>•<br>Last Price and Offering Price must be able to<br>accept and store up to 5 decimal digits.|RR=0.473;<br>P@K=1.000;<br>MRR=1.000;<br>Overall=0.304;<br>KTE=0.509|



 15 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|||•<br>Input values (e.g., 30.12345) must be stored<br>exactly as entered (30.12345), rather than<br>being rounded or truncated to 2 decimal<br>digits (e.g., 30.12).<br>This change will not be affected by the<br>offering parameter.<br>Previously, there was a table-level validation<br>limiting input to only two decimal digits, which<br>DEV_B referred to as a "hardcode." However,<br>after confirmation from PO_1 with the user, it<br>was agreed to change this to accommodate the<br>user's requirements.||
|---|---|---|---|
|5|Apa status implementasi<br>fitur amend pada modul<br>trade custody<br>berdasarkan diskusi tim<br>Februari 2025 ?<br>_What is the_<br>_implementation status of_<br>_the amend feature in the_<br>_trade custody module,_<br>_based on the team's_<br>_February 2025_<br>_discussions?_|The answer identifies the following,<br>Based on the team's discussions in February 2025,<br>the amend feature in the trade custody module is<br>currently in the development and testing stage.<br>Several points regarding its implementation status<br>are as follows:<br>-  There was a discussion on the "confirm amend"<br>behavior in trade custody, where the second<br>confirmation (confirm amend) does not create a<br>new row or convert the previous one to inactive,<br>unlike the first confirmation. (09/02/2025)<br>- PO_3 stated that the value in the amend should<br>be recalculated if there is a change. (12/02/2025)<br> <br>Ini This indicates that the feature is still<br>undergoing development, testing, and refinement,<br>with several functional and display issues<br>currently being addressed.|RR=0.474;<br>P@K=1.000;<br>MRR=1.000;<br>Overall=0.293;<br>KTE=0.476|



## **5. Cross-Scenario Prompt Level Highlights** 

To provide granular insight beyond scenario-level means, Table 10 presents representative promptlevel results from Scenarios B-E, selecting the highest-scoring and lowest-scoring question per scenario to illustrate performance variance. Several observations emerge from the prompt-level analysis: 

 16 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

**Table 10. Selected prompt-level results across scenarios (best and worst per scenario)** 

|**_Scenario_**|**_No._**|**_Prompt_**|**_RR_**|**_Faith_**|**_Comp_**|**_KTE_**|**_Overall_**|
|---|---|---|---|---|---|---|---|
|B (Best)|B2|Apa perbedaan antara sesi<br>General dan sesi Restricted<br>dalam sisstem BOND_SYS ?<br>_What is the difference_<br>_between the General session_<br>_and the Restricted session in_<br>_the BOND_SYS system?_|0.526|0.179|0.875|0.527|0.333|
|B (Worst)|B5|Bagaimana mekanisme<br>pengiriman notifikasi<br>broadcast kepada peserta<br>setelah alokasi ?<br>_What is the mechanism for_<br>_sending broadcast_<br>_notifications to participants_<br>_after allocation?_|0.613|0.104|0.375|0.240|0.228|
|C (Best)|C5|Dari semua quotation pada<br>RFQ 20140327-01, berapa<br>yang dialokasikan ?<br>_Of all the quotations under_<br>_RFQ 20140327-01, how many_<br>_were allocated?_|0.565|0.030|0.857|0.444|0.293|
|C (Worst)|C3|Apa perbedaan auction_unit<br>antara board BS-SB<br>dibandingkan board lainnya ?<br>_What is the difference in_<br>_auction_unit between the BS-_<br>_SB board and other boards?_|0.277|0.050|0.167|0.109|0.103|
|D (Best)|D4|Berdasarkan FR dan data<br>aktual, bagaimana<br>settlement_date ditetapkan ?|0.676|0.185|0.857|0.521|0.359|



 17 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|||_Based on the FR and actual_<br>_data, how is the_<br>_settlement_date determined?_||||||
|---|---|---|---|---|---|---|---|
|D (Worst)|D1|Apakah jam sesi pada data<br>RFQ aktual sudah sesuai<br>spesifikasi?<br>_Does the session time in the_<br>_actual RFQ data comply with_<br>_the specification?_|0.534|0.121|0.200|0.161|0.178|
|E (Best)|E2|Apa keputusan teknis terkait<br>offering digit dan validasi<br>hardcode yang didiskusikan<br>oleh tim ?<br>_What technical decision_<br>_regarding the offering digit_<br>_and hardcode validation was_<br>_discussed by the team?_|0.588|0.112|0.900|0.506|0.428|
|E (Worst)|E3|Berdasarkan log diskusi tim,<br>apa yang terjadi pada insiden<br>ETL BOND_SYS ?<br>_Based on the team's_<br>_discussion log, what_<br>_happened during the_<br>_BOND_SYS ETL incident?_|0.491|0.047|0.667|0.357|0.308|



## **6. Error Analysis and Failure Cases** 

A systematic examination of failure cases reveals three distinct failure patterns in the system's 

responses: 

Pattern 1: Complete Retrieval Failure ("Information Not Found"). Question A1 ("Apa masalah yang ditemukan tim saat proses submit quotation pada board BS-SB menjelang demo, dan bagaimana workaround sementara yang disepakati?") produced a response explicitly stating: "Informasi tidak ditemukan dalam sumber data yang tersedia. Konteks tidak menyebutkan masalah yang ditemukan tim saat proses submit quotation pada board BS-SB menjelang demo." Despite achieving P@K = 1.000 and RR = 0.441, the generative model determined that the retrieved chunks did not contain sufficient evidence to answer the question. This represents a conservative behaviour where the model refuses to hallucinate rather than generating an unsupported answer. Notably, Answer Completeness remained high (0.867) because the response echoes question keywords, while Answer Faithfulness was very low (0.031) due to the absence of substantive content grounded in the context. 

 18 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

Pattern 2: Low Completeness from Structured Data. Question C3 achieved the lowest Overall score across all 25 questions (0.103) with Answer Completeness = 0.167. The system correctly retrieved and reported the auction_unit values per board (BS-SB = "Mio", BS = "Bio", BC = "Bio"), but the answer was concise and factual rather than explanatory. Token-overlap metrics penalize short, correct answers because they lack the verbose explanation needed to cover question keywords like "perbedaan" (difference). This represents a metric limitation rather than a system failure: the answer is factually correct but scores poorly on Completeness because the evaluation metric rewards keyword coverage over factual accuracy. 

Pattern 3: Partial Answer with Cross-Reference Gap. Question D1 (Overall = 0.178, Completeness = 0.200) required comparing session times from FR specification documents against actual RFQ data in the database. The system retrieved relevant chunks from both sources but generated only a partial answer covering the specification side without explicitly cross-referencing the database values. This reveals a limitation in the generative model's ability to perform explicit comparison across heterogeneous source types within a single answer, even when relevant contexts from both sources are successfully retrieved. 

Summary of failure distribution: Of 25 questions evaluated, 1 produced a complete retrieval failure (Pattern 1), 3 achieved Overall < 0.200 due to low Completeness from structured/comparative queries (Pattern 2), and 2 exhibited partial cross-reference gaps (Pattern 3). The remaining 19 questions (76%) achieved Overall ≥ 0.228 with substantive answers covering the majority of question intent. These patterns suggest that future improvements should prioritize: (a) enhanced prompt engineering for comparative reasoning, (b) answer expansion strategies for structured data queries, and (c) explicit cross-source synthesis instructions in the generation prompt. 

## **Visualization of Results** 

Evaluation results are visualized across four separate panels (Figures 3a-3d): 

Figure 3a presents the mean standard metrics per scenario in a grouped bar chart. P@K and MRR consistently reach 1.000 across all scenarios, indicating that the retrieval component performs optimally regardless of source type configuration. This ceiling effect in retrieval metrics confirms that the combination of multilingual MiniLM-L12-v2 embeddings and FAISS IndexFlatL2 provides sufficient discriminative power for the 86-chunk corpus. Scenario B exhibits the highest ROUGE-L among referencefree scenarios (0.051), while Scenario E achieves ROUGE-L = 0.181 and BLEU-1 = 0.196 (referencebased). The substantial gap between reference-free and reference-based ROUGE-L values (0.051 vs. 0.181) suggests that reference-free ROUGE-L systematically underestimates generation quality, as it compares against retrieved chunks rather than ideal answers. 

Figure 3b presents the overall score per question (25 questions across 5 scenarios), revealing the distribution of performance across questions and scenarios. A notable observation is the variance within scenarios: Scenario A exhibits relatively consistent scores (range: 0.269–0.304), while Scenario C shows higher variance (range: 0.165–0.289), indicating that structured database content is more sensitive to question formulation. Questions requiring numeric lookup from PostgreSQL tables (e.g., "default price percentage values") achieve lower Overall scores than questions requiring narrative synthesis, because token overlap metrics (Faithfulness, ROUGE-L) penalize concise numeric answers that are factually correct but lexically dissimilar from the context. 

 19 

**eISSN 3063-802X & pISSN 3063-8011** _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ LD ! | **(Adi, et al.)** JURNAL RAGAM PENGABDIAN - PENELITIAN 

Figure 3c compares three composite metrics (KTE, MSRS, AQI) across scenarios. MSRS is highest in Scenario C (PostgreSQL, 0.825), reflecting that the 8-table corpus provides high source diversity in topK retrieval. Conversely, Scenario C records the lowest KTE (0.312), revealing a trade-off: high source diversity does not guarantee effective knowledge transfer when the source content is structured data lacking narrative explanation. Scenario A (Chat Only) achieves the highest KTE (0.464) despite relatively low MSRS (0.725), suggesting that conversational knowledge, while concentrated in fewer sources, provides richer contextual information for operational question answering. Scenario E yields KTE = 0.453 despite its questions being the most complex, demonstrating that multi-source integration compensates for perlayer limitations. 

Figure 3d is a radar chart illustrating the multi-dimensional profiles of all five scenarios, confirming that each scenario exhibits a distinct and complementary profile: A leads in Completeness (0.853) and also leads among reference-free scenarios in Overall (0.285) and KTE (0.464), indicating conversational data covers question keywords comprehensively and supports effective transfer; B leads in Faithfulness (0.114) among single-source scenarios, showing formal documents provide more faithful grounding; C leads in MSRS (0.825) due to source diversity from 8 tables; and E leads in Retrieval Relevance (0.582) and reference-based metrics. The radar chart visually confirms that no single scenario dominates all dimensions simultaneously, validating the multi-scenario evaluation design. 

**Figure 3a** . Mean standard metrics per scenario (grouped bar chart) 

 O 20 20 

**eISSN 3063-802X & pISSN 3063-8011** 

**==> picture [462 x 36] intentionally omitted <==**

**----- Start of picture text -----**<br>
Agnostic Multi-Source Retrieval-Augmented Generation for Documents and<br>Database Question Answering<br>(Adi, et al.)<br>JURNAL RAGAM PENGABDIAN - PENELITIAN<br>**----- End of picture text -----**<br>


**Figure 3b.** Overall score per question (25 questions, 5 scenarios) 

**Figure 3c.** Composite metrics KTE, MSRS and AQI per scenario 

 O 21 21 

**eISSN 3063-802X & pISSN 3063-8011** _**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** Gi ~~JURAGAN~~ JURNAL RAGAM PENGABDIAN - PENELITIAN 

**Figure 3d.** Radar chart of multi-dimensional profiles across five scenarios 

## **7. Comparison with Existing Multi-Source and Adaptive RAG Approaches** 

To position the contribution of this study relative to prior and concurrent work, Table 11 compares the proposed agnostic multi-source architecture with classical RAG (Lewis et al. 2020) and two recent (2025) heterogeneous/adaptive RAG systems identified in the literature: HetaRAG (Yan et al. 2025), which orchestrates retrieval across heterogeneous data stores, and HyPA-RAG (Kalra et al. 2025), a hybrid parameter-adaptive RAG system for high-stakes document QA. 

**Table 11. Qualitative comparison with existing RAG/QA approaches** 

|**_Aspect_**|**_This Study_**|**_Classical RAG (Lewis et al. 2020)_**|**_HetaRAG (Yan et al. 2025)_**|**_HyPA-RAG (Kalra et al. 2025)_**|
|---|---|---|---|---|
|Source types|Unstructured documents (PDF/TXT) and relational database (PostgreSQL), unified via a single source parameter|Single unstructured text corpus|Vector index, knowledge graph, full-text engine, and structured database, orchestrated in parallel|Primarily unstructured legal/policy documents|
|Indexing strategy|Real-time, in-memory FAISS index built per invocation, with session-level caching|Pre-indexed, static corpus|Multiple persistent specialized stores (vector, graph, full-text, SQL)|Pre-indexed, static corpus|
|Source extensibility|Adapter Pattern (FolderSourceAdapter, PostgreSQLAdapter, MultiSourceAdapter); new sources added via two methods without modifying the core pipeline|Not source-agnostic; retrieval pipeline is tied to the corpus format|Extensible but requires provisioning and maintaining an additional specialized store per data type|Not source-agnostic; tuned for a single document domain|
|Adaptivity mechanism|Uniform pipeline across sources; adaptivity is achieved by combining layers (L1/L2/L3) rather than per-query parameter tuning|None|Cross-store orchestration logic per query|Query-complexity classifier that adapts retrieval parameters (e.g., top-K) per query|
|Evaluation approach|8 metrics plus 3 purpose-built composite scores (KTE, MSRS, AQI) across 5 progressive multi-source scenarios and an ablation study|Standard QA benchmarks (e.g., Natural Questions, TriviaQA)|Domain benchmark evaluation of end-to-end retrieval-generation accuracy|Legal/policy QA accuracy and retrieval-cost trade-off metrics|
|Application domain|Organizational knowledge transfer in a document-based financial service platform|Open-domain QA|General heterogeneous enterprise knowledge|AI legal and policy applications|

Relative to classical single-source RAG, this study contributes an explicit source-agnostic abstraction (SourceDetector + SourceFactory + Adapter Pattern) and a composite evaluation framework tailored to knowledge transfer rather than open-domain QA accuracy alone. Compared with HetaRAG, which achieves heterogeneity by orchestrating multiple persistent specialized stores, this study achieves a comparable multi-source capability with a single lightweight in-memory FAISS index rebuilt at query time, trading multi-hop cross-store reasoning for lower infrastructure overhead and always-current content. Compared with HyPA-RAG, which adapts retrieval parameters per query within a single document domain, this study instead adapts at the source-configuration level — that is, in terms of which adapters are active. The ablation study (Table 6) then provides direct empirical evidence, rather than only architectural description, of how much each additional source layer contributes to answer quality: the Overall score rises from 0.214-0.237 for single layers to 0.373 once all three layers are combined. Taken together, these comparisons indicate that this study's specific niche, a lightweight, real-time, adapter-based agnostic RAG evaluated with knowledge-transfer-oriented composite metrics, is not directly covered by existing heterogeneous or adaptive RAG systems in the literature. 

## **Result Limitation** 

The retrieval component uses paraphrase multilingual-MiniLM-L12-v2 (50+ languages), ensuring no language bias at the retrieval layer; retrieval weights are entirely determined by semantic similarity, not source type metadata. For single-entity questions, Context Coverage tends to be low because top-K chunks concentrate on a single document; this accurately reflects the corpus structure rather than a system bias. Content hash-based deduplication is implemented to prevent a single file from being counted as multiple distinct sources. 

Several limitations of this study must be explicitly acknowledged: 

- Evaluation coverage. The evaluation covers 25 questions across five scenarios within a single domain (a domain-specific financial instrument transaction platform, BOND_SYS) using an Indonesian-language corpus. Empirical validation across other domains and languages is required to strengthen the generalizability of the results. 

- Partial ground truth. Only Scenario E (5 questions) is equipped with manually curated reference answers (ground truth). Scenarios A-D employ reference-free evaluation, such that ROUGE-L and BLEU-1 for those four scenarios are computed against retrieved context rather than ideal reference answers. Scenario E’s reference-based ROUGE-L (0.181) and BLEU-1 (0.196) 

 O 22 22 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

cannot be directly compared with the reference-free ROUGE-L of Scenarios A-D (0.0220.046). Extending ground truth to all 25 questions would improve comparative validity across scenarios. 

- Evaluation scale. Evaluation across 25 questions in five scenarios provides an adequate proofof-concept, but is insufficient for statistically generalizable claims. Future research is recommended to employ at least 20-50 query-answer pairs per scenario with curated ground truth across all scenarios. 

- LLM non-determinism. The generative component (Gemini 2.5-flash) is inherently nondeterministic: identical prompts may produce slightly different answers across runs, potentially affecting token-overlap metrics (Faithfulness, ROUGE-L, BLEU-1). All reported results are from a single evaluation run; repeated trials with statistical aggregation would strengthen the reliability of generation-side metrics. 

- Composite metric validity. The three composite metrics proposed in this study (KTE, MSRS, AQI) are author-defined formulations designed for multi-source RAG evaluation. While each component metric is grounded in established IR and NLP literature, the specific composite formulations have not been independently validated by the research community. Their applicability beyond the evaluation context of this study should be verified through replication in other multi-source RAG settings. 

## **Implication** 

## **1. Implication for Organizational Knowledge Transfer** 

The system directly addresses the challenge identified in the introduction. A newly onboarded Product Owner can immediately pose questions in natural language, for example, “What bug was found in the submit quotation on board BOARD_TYPE_A and how was it resolved?” or “What is the implementation status of the amend feature in the BOND_MOD_CUSTODY?” , and the system will automatically search for answers from the combination of PDF system specification documents, 908message team discussion logs and the BOND_SYS PostgreSQL database within a single query. This scenario represents knowledge transfer from previous personnel to their successors without requiring a complete re-reading of all scattered documentation, consistent with the challenge of tacit knowledge explicitation identified by Nonaka and Takeuchi (1995) and with knowledge management system design requirements highlighted by Alavi and Leidner (2001). The same principle applies to other organizational contexts: new analysts can query technical decisions or system configurations from existing project documentation. 

## **2. Implication for System Extensibility** 

The Adapter Pattern architecture opens a path for extending to new source types. MongoDB, SharePoint and Google Drive API can each be integrated by implementing two methods, load() and describe(), without modifying the core pipeline. This aligns with the Open/Close Principle in software design (Martin 2017). The real-time vs. pre-indexed trade-off should be considered according to use case, as summarised in Table 12. 

## **Table 12. Trade-off Comparison: Real-Time vs. Pre-Indexed Indexing** 

 23 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

|**_Aspect_**|**_Real-time (this system)_**|**_Pre-indexed_**|
|---|---|---|
|Data consistency|Always up-to-date|Potentially stale|
|Cold start time|~2-5 seconds|Instan|
|Storage management|No disk overhead|Requires synchronization|
|Best suited for|Frequently changing documents|Large static corpora|



For very large static corpora (millions of documents), FAISS IVF (Inverted File Index) or hybrid search integration with BM25-based hybrid search can be considered as architectural evolution (Roberstson and Zaragoza 2009; Manning et al. 2008). 

## **3. Implications for RAG Evaluation Methodology** 

The evaluation findings of this study reveal several methodological insights applicable to future RAG system evaluations. First, the consistent Precision@K = 1.000 and MRR = 1.000 across all 25 questions demonstrates that for small-to-medium corpora (< 100 chunks), FAISS IndexFlatL2 with appropriate embedding models achieves perfect retrieval, rendering retrieval metrics uninformative as discriminators. Future evaluations on similar corpus scales should prioritize generation of quality metrics and composite metrics that capture answer utility rather than retrieval precision alone. 

Second, the divergence between reference-free and reference-based ROUGE-L (0.022–0.046 vs. 0.181) highlights a fundamental measurement of asymmetry. Reference-free ROUGE-L, computed against retrieved context, penalizes abstractive generation models that synthesize information rather than extracting verbatim. This observation suggests that reference-free evaluation is better suited as a faithfulness proxy (measuring extractive alignment) rather than as a quality indicator, a distinction that should be explicitly stated in future RAG evaluations lacking ground truth. 

Third, the composite metrics proposed in this study (KTE, MSRS, AQI) address evaluation gaps not covered by individual metrics. KTE captures whether knowledge transfer succeeds from both antihallucination and coverage perspectives simultaneously. MSRS prevents single-source retrieval from being misinterpreted as multi-source capability. AQI bridges content adequacy with linguistic quality. These composite formulations provide a reusable evaluation template for other multi-source RAG implementations in organizational contexts, particularly where evaluation resources are limited and full ground truth annotation is impractical. 

Fourth, the partial ground truth strategy employed in this study, where only the most complex scenario (E) receives curated reference answers, represents a pragmatic evaluation design for resourceconstrained research settings. This strategy concentrates annotation effort on the scenario most likely to reveal system limitations while using reference-free metrics as lower-bound estimates for simpler scenarios. Future studies may adopt this graduated approach when full-corpus annotation is prohibitively expensive. 

## **CONCLUSION** 

 24 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

This study successfully develops an agnostic multi-source RAG system fulfilling the research objectives, with six principal findings: 

1. Agnostic architecture realized: SourceDetector + SourceFactory + BaseSourceAdapter enable handling of folders (TXT/PDF) and PostgreSQL (3 query modes) from a single source parameter, without modifications to the core pipeline. Automatic source type detection via pattern matching operates consistently across all 25 questions. 

2. Real-time indexing confirmed: Each invocation of pipeline.ask() constructs the FAISS index in-memory with no index files on disk. Content changes in the source are immediately reflected in retrieval results without system restart. 

3. Multi-source empirically proven: Batch evaluation of 25 questions (5 scenarios x 5 questions) shows Precision@K = 1.000 and MRR = 1.000 across all scenarios. Aggregate performance: A (Chat, Overall = 0.285, MSRS = 0.725), B (PDF, Overall = 0.284, MSRS = 0.625), C (PostgreSQL, Overall = 0.221, MSRS = 0.825), D (PDF + DB, Overall = 0.279, MSRS = 0.763), E (Hybrid, Overall = 0.373, MSRS = 0.713). Scenario E proves the source-agnostic claim at the highest level: Retrieval Relevance = 0.582, the highest among all scenarios, confirming that a single query can reach relevant chunks from all three corpus layers simultaneously. 

4. Ground truth Scenario E yields meaningful ROUGE-L and BLEU-1: With 5 manually curated reference answers, Scenario E achieves ROUGE-L = 0.181 and BLEU-1 = 0.196 (referencebased), demonstrating genuine content overlap between AI-generated answers and ideal answers for cross-layer questions. This represents approximately a ~4x improvement over the highest reference-free ROUGE-L (Scenario B, ROUGE-L = 0.046). 

5. Ablation study proves dramatic Layer 3 contribution: As detailed in 4.1.3, Overall scores remain relatively similar across Chat-only, PDF-only, and PDF + DB configurations (0.237, 0.214, 0.234 respectively), but jump sharply to 0.373 when all three layers are activated (+0.139 over PDF+DB). ROUGE-L increases from 0.051 to 0.181 and BLEU-1 from 0.002 to 0.196, proving that team discussion logs (Layer 3) are the determining component for cross-paradigm answer quality. 

6. Knowledge transfer effectiveness measured: KTE per scenario: A = 0.464 (tacit to operational), B = 0.395 (explicit to actionable), C = 0.312 (explicit to structured), D = 0.389 (explicit to cross-referenced), E = 0.453 (cross-paradigm). Scenario A achieves the highest KTE (0.464), followed by E (0.453) and B (0.395), demonstrating that conversational knowledge sources and multi-source integration consistently produce effective knowledge transfer. 

The principal contributions of this study are: (i) an Adapter Pattern design that separates concern among data sources, text processing, retrieval, and generation; (ii) a five-scenario evaluation design with a three-layer real operational corpus; (iii) a partial ground truth framework that enables reference-based validation on priority scenarios without requiring complete ground truth across all scenarios; and (iv) a positioning of this contribution relative to existing multi-source and adaptive RAG systems (Table 11) and to traditional retrieval, supported empirically by two baselines. A zero-shot LLM baseline (Table 7) failed to answer 92% of the evaluation questions from parametric knowledge alone (Overall = 0.029 vs. 0.373 for Scenario E), confirming that the reported gains are attributable to the retrieval-grounded architecture rather than to the underlying LLM's pretrained knowledge. A BM25 keyword-search baseline (Table 8) was competitive with dense semantic retrieval on single-source scenarios but was clearly outperformed on the cross-referencing Scenario D (Overall 0.279 vs. 0.239), indicating that dense retrieval's empirical value is concentrated in exactly the multi-source, cross-referencing capability this study is designed to provide, rather than being uniformly superior across all retrieval conditions. 

For future research, the following directions are recommended: (1) extending ground truth to all 25 questions for stronger comparative validation; (2) hybrid search (FAISS + BM25) to improve retrieval on queries with domain-specific tokens; (3) fine-tuning the embedding model on organization-specific domain 

 25 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

documents; and (4) comparison with commercial vector database-based RAG systems (Pinecone, Weaviate) as architectural baselines. 

## **REFERENCE** 

- Chui, M, Manyika J, Bughin J, Dobbs R, Roxburgh C, Sarrazin H, Sands G, Westergreen M (2012) The social economyL unlocking value and productivity through social technologies. McKinsey Global InstituteFriends and neighbors on the web. Soc Networks 25(3):211–230. https://www.mckinsey.com/industries/technology-media-and-telecommunications/ourinsights/the-social-economy 

- Alavi MJ, Leidner DE (2001) Review: knowledge management and knowledge management system. MIS Q 25(1): 107-136. https://doi.org/10.2307/3250961 

- Borgeaud S, Mensch A, Hoffman J, Cal T, Rutherford E, Millican K, van den Driessche g, Lespiau JB, Damoc B, Clark A, et al. (2022) Improving language models by retrieving from trillions of tokens. In: Proceedings of the 39th International Conference on Machine Learning (ICML). https://doi.org/10.48550/arXiv.2112.04426 

- Carbonell J, Goldstein J (1998) The use of MMR, diversity-based reranking for reordering documents and producing summaries. In: Proceedings of the 21st Annual Annual I nternational ACM SIGIR Conference on Research and Development in Information Retrieval, pp 335-336. https://doi.org/10.1145/290941.291025 

- Cheng M, Luo Y, Ouyang J, Liu Q, Liu H, Li L, Yu S, Zhang B, Cao J, Ma J, Wang D, Chen E (2025) A survey on knowledge-oriented retrieval-augmented generation. arXiv:2503.10677. https://doi.org/10.48550/arXiv.2503.10677 

- Es s, James J, Espinosa-Anke L, Schockaert S (2023) RAGAS: automated evaluation of retrieval augmented generation. arXiv:2309.15217. https://doi.org/10.48550/arXiv.2309.15217 

- Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazare PE, Lomeli M, Joulin A, Jegou H (2024) The FAISS library. arXiv:2401.08281. https://doi.org/10.48550/arXiv.2401.08281 

- Fabbri AR, Krycinski W, McCann B, Xiong C, Socher R, Radev D (2021) SummEval: re-evaluating summarization evaluation. Trans Assoc Comput Linguist 9:391-409. https://doi.org/10.1162/tacl_a_00373 

- Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, Dai Y, Sun J, Wang M, Wang H (2024) Retrieval-augmented generation for large language models: a survey. arXiv:2312.1097. https://doi.org/10.48550/arXiv.2312.10997 

- Gamma E, Helm R, Johnson R, Vlissides J (1994) Design patterns: elements of reusable object-oriented software. Addison-Wesley. https://books.google.co.id/books?id=6oHuKQe3TjQC&printsec=frontcover&source=gbs_ge_su mmary_r&cad=0#v=onepage&q&f=false 

- Gartner (2020) Gartner says employees spend too much time on low-value tasks: use AI and automation to fix it. Gartner Newsroom. https://www.gartner.com/en/newsroom/press-releases/2020-01-23gartner-says-employees-spend-too-much-time-on-low-value-tasks 

- IDC (2023) 90% of data is unstructured and it’s full of untapped value. IDC Blog. https://blogs.idc.com/2023/05/09/90-of-data-is-unstructured-and-its-full-of-untapped-value/ 

- Izacard G, Grave E (2021) Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pp 874-880. https://doi.org/10.18653/v1/2021.eacl-main.74 

- Kalra R, Wu Z, Gulley A, Hilliard A, Guan X, Koshiyama A, Treleaven PC (2025) HyPA-RAG: a hybrid parameter adaptive retrieval-augmented generation system for AI legal and policy applications. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Industry Track, pp 1036-1054. https://doi.org/10.18653/v1/2025.naacl-industry.79 

 26 

**eISSN 3063-802X & pISSN 3063-8011** 

_**Agnostic Multi-Source Retrieval-Augmented Generation for Documents and Database Question Answering**_ **(Adi, et al.)** 

Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, Ishil E, Bang YJ, Mandotto A, Fung P (2023) Survey of hallucination in natural language generation. ACM Comput Surv 55(12):1-38. https://doi.org/10.1145/3571730 

- Johnson J, Douze M, Jegou H (2019) Billion-scale similarity search with GPUs. IEEE trans Big Data 7 (3):535-547. https://doi.org/10.1109/TBDATA.2019.2921572 

- Karpukhin V, Oguz B, Min S, Lewis P, Wu L, Edunov S, Chen D, Yih W (2020) Dense passage retrieval for open-domain question answering. In: Proceedings of EMNLP 2020, pp 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550 

- Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kiela D (2020) Retrieval-Augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems 33, pp 9459-9474. https://doi.org/10.48550/arXiv.2005.11401 

- Lin CY (2004) ROUGE: a package for automatic evaluation of summaries. In: Proceedings of the ACL Workshop on Text Summarization Branches Out. https://aclanthology.org/W04-1013 

- Manning CD, Raghavan P, Schutze H (2008) Introduction to information retrieval. Cambridge University Press, Cambridge. 

- Martin RC (2017) Clean architecture: a craftsman’s guide to software structure and design. Prentice Hall, Upper Saddle River 

Nonaka I, Takeuchi H (1995) the knowledge-creating company. Oxford University Press, New York. 

- Papineni K, Roukos S, Ward T, Zhu WJ (2002) BLEU: a method for automatic evaluation of machine 

   - translation. In: Proceedings of the 40th Annual Meeting of Association for Computational Linguistics, pp 311-318. https://doi.org/10.3115/1073083.1073135 

- Robertson S, Zaragoza H (2009) The probabilistic relevance framework: BM25 and beyond. Found Trends Inf Retr 3(4):333-389. https://doi.org/10.1561/1500000019 

- Reimers N, Gurevych I (2019) Sentence-BERT: sentence embeddings using Siamese BERT-networks. IN: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. https://doi.org/10.18653/v1/D19-1410 

- Asai A, Wu Z, Wang Y, Sil A, Hajishirzi H (2023) Self-RAG: learning to retrieve, generate, and critique through self-reflection. arXiv:2310.11511. https://doi.org/10.48550/arXiv.2310.11511 

- Sharma C (2025) Retrieval-augmented generation: a comprehensive survey of architectures, enhancements, and robustness frontiers. arXiv:2506.00054. https://doi.org/10.48550/arXiv.2506.00054 

- Shi P, Li X, Han X, Chang B, Sui Z (2023) REPLUG: retrieval-augmented black-box language models. arXic:2301.12652. https://doi.org/10.48550/arXiv.2301.12652 . 

- Voorhees EM (1999) The TREC-8 question answering track report. In: Proceedings of the 8th Text retrieval Conference (TREC-8), pp 77-82. https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf . 

- Yan G, Zhang Y, Cai P, Wang D, Mao S, Zhang H, Zhang Y, Zhang H, Cai X, Shi B (2025) HetaRAG: hybrid deep retrieval-augmented generation across heterogeneous data stores. arXiv:2509.21336. https://doi.org/10.48550/arXiv.2509.21336 

- Yasunaga M, Ren H, Bosselut A, Liang P, Leskovec J (2021) QA-GNN: reasoning with language models and knowledge graphs for question answering. In: Proceedings of the 2021 Conference of the North American Chapter of the association for Computational Linguistics, pp 535-545. https://doi.org/10.18653/v1/2021.naacl-main.45 

 27 

