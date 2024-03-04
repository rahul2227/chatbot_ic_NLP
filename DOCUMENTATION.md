# BioMedical Question Answering System Based on PubMed Abstracts.

### Team Members

- Rahul Sharma (rahul.sharma@stud.uni-heidelberg,de)
  - Matrikulation number - 3773074
  - Data and Computer Science (M. Sc.)
- Vasu Thakur (vasu.thakur@stud.uni-heidelberg,de)
  - Matrikulation number - 3773479
  - Data and Computer Science (M. Sc.)
- Vikas Kumar Roy (vikas.roy@stud.uni-heidelberg,de)
  - Matrikulation number - 3773130
  - Data and Computer Science (M. Sc.)

## Member Contribution

Rahul Sharma - Contributed in creating query transformation and Multiquery retrieval features and online setup for the project. Contributed in model analysis of turbo GPT 3.5 and gptneo, Created the command line interface for pipelines and automations. Model Evaluations

Vasu Thakur - Contributed in creating query transformation and Multiquery retrieval features and online setup for the project. Created the command line interface for pipelines and automations. Contributed in model analysis of turbo GPT 3.5 and galactica, Model Evaluations

Vikas Kumar Roy - Created embeddings for vector database online, Created initial scripts for querying the data from pubmed, contributed in model analysis for BioMedLM and galactica

### Advisor

- Michael Gertz
- Robin Khanna

## Introduction

Accurate and timely information retrieval is critical in the continuously changing field of biomedical research. The exponential growth of published literature is a challenge to academics and healthcare professionals who must search through huge knowledge libraries to uncover pertinent information. In order to address this difficulty, this study presents Cellio, an LLM RAG (Retrieval-Augmented Generation) system that uses PubMed abstract analysis to provide precise responses to biomedical queries.

Based on a vector store retriever, Cellio is a RAG system that offers query transformed context retrieval to enhance both the density and quality of retrieved data. With this method, we can improve the responses by answering user queries with more pertinent context than we could have otherwise.

This study is organized to give a comprehensive knowledge of Cellio. It begins with a summary of relevant work that places our RAG in relation to current approaches. After that, it explores the techniques that support Cellio, going into detail about its architecture and component integration. The system's capabilities are demonstrated and its performance is compared to pertinent metrics in the section on the experimental setup and outcomes. The project's contributions to the field of biological information retrieval are finally summed up in the conclusion, which also suggests possible directions for further study.

## Related Work

The recently announced GPT-4 and GT-3.5-turbo generative pre-trained transformer models are pre-trained on the next token prediction task and are based on the transformer architecture [[1]](#1). These models are further refined by reinforcement learning based on human input, which significantly enhances their capacity to adhere to instructions and the perceived usefulness of their offspring.

Retrieval-augmented generation (RAG) models are presented by Lewis et al. (2020). RAG models integrate parametric and non-parametric memory that has been trained beforehand to generate language. The Wikipedia dense vector index serves as the non-parametric memory in RAG models, which are accessed by a pre-trained neural retriever using a pre-trained seq2seq model as the parametric memory. The authors compare two RAG model variations: one that can use multiple passages for each token and one that uses the same retrieved passages for the whole generated sequence. The study demonstrates that RAG models produce more factual and diversified language than a seq2seq baseline and obtain state-of-the-art outcomes on numerous knowledge-intensive NLP tasks, including open-domain QA and fact verification. [[2]](#2)

Ma et al. (2023) present a novel query rewriting-based framework for retrieval-augmented large language models (LLMs). The concept is to use a web search engine and a trainable rewriter to adapt the input query to the frozen modules of the LLM reader. With the help of the LLM reader's feedback, a little language model known as the rewriter is refined through reinforcement learning. The authors test their framework using multiple-choice and open-domain QA tasks, and they consistently outperform the baseline retrieve-then-read method in terms of performance. The study presents a fresh viewpoint and a workable fix for retrieval-augmented LLMs. [[3]](#3)

## Methods/Approaches

![NLP Query flow.png](media/NLP%20Query%20flow.png)

Figure 1: a simplified architecture of our RAG System
While Our architecture optimizes retrieval in a few crucial aspects, it essentially adheres to the RAG model proposed by *Lewis et al. 2*̥. The abstract, title, authors, keywords (if available), and publication date were first scraped from the PubMed abstracts (the appropriate code can be found in `lib/backend/data/data-aqui.ipynb`). Following chunking and embedding using the `all-MiniLM-L12-v2` model, the abstracts from the approximately ~66k collected documents were saved in an Chroma vector database (the pertinent code is available in `lib/backend/data/database-functions/chroma_query.ipynb`).

The process illustrated in Fig. 1 would be followed by a user request to our model (in {askme.py}). Initially, the query is sent straight to our Vector DB, which specializes in retrieving queries using the cosine similarity index. The context is then provided to our multi-query retrieval pipeline (located in {askme.py}), which also acts as a query transformer for our task, after we receive pertinent context from our vector database. In order to account for the previous context and query, the Query transformer employs a customized system query. After embedding our questions using the `all-MiniLM-L12-v2`} model, it searches the chroma vector database for similarity and selects the three most comparable abstracts. After that, the context is concatenated and fed into our language.

We chose this particular architecture because it was both easily implemented and produced results of high quality. Since vector stores allow for quick retrieval of unstructured text, we opted to use them rather than something like graph databases. Chromadb was chosen as he vector storage solution as it provided great documentation for retrieval performance and DB hosting, its easy code implementation also became a considerable factor in choosing this database. We used `all-MiniLM-L12-v2` as our embedding model due to its lightweight and its ability to be run on CPU because of that fact, it also delivers excellent performance in comparison to its size. Turbo GPT 3.5 was chosen as our generational model as it performed excellently among a pool of AI models that we selected from huggingface leaderboard and community papers. Our method uses the abstract chunks' embedding vectors to first discover generally related content, and then compression to fine-tune the context. We seldom ever encountered sentences that contradicted one another around the retrieved portions because the entire retrieved text is of a scientific character. Because of this, we choose not to employ an overlap while creating our retrieved sentences. In order to obtain context-relevant results that were not specifically mentioned in the initial query, we decided to use a query transformer to produce retrieval requests. The use of two pipelines allows for precision when retrieving documents from the vector store whilst provoking well written and diverse responses. This pipeline make sure to get the context relevant to user query, however, for complex user query the pipeline  still allow for variation to find related articles not explicitly named in the user query.

### Python Libraries

- ChromaDB - Used for vector storage access.
- Transformers - Used to run/load hf models, and embed documents.
- Langchain - Used to chain relevant LLM functions together.
- HuggingFace - Used to download / load models
- BitsAndBytes - Used to quantize the model
- BeautifulSoup - used to parse html content for ~~stealing~~ extracting the abstracts
- tqdm - Helped monitor the progress of loops

## Experimental Setup and Results

To assess the effectiveness of our framework, we formulated 60 evaluation inquiries spanning six distinct domains:

1. _Yes/No:_ Queries answerable with a simple "yes" or "no."
2. _Factual:_ Inquiries seeking specific factual information, typically beginning with "what," "where," or "when."
3. _List:_ Questions soliciting a list, such as the top contributing factors to a particular phenomenon.
4. _Causal:_ Inquiries probing for causal relationships, typically initiated with "how" or "why."
5. _Hypothetical:_ Queries exploring hypothetical scenarios, often phrased as "what would happen if..."
6. _Complex:_ Intricate questions delving into the interplay between two or more concepts.

Both the GPT-3.5-turbo and GPT-4-1106-preview models were tasked with responding to all 60 questions. Subsequently, we utilized the GPT-Eval framework to assess the quality of their answers. This framework systematically instructs ChatGPT 4 to evaluate the provided responses based on various criteria, including grammar, coherence within the answer, relevance and utilization of context, and overall accuracy of the statement. Notably, this evaluation methodology closely aligns with human assessments, contrasting with the stringent nature of many primary benchmarks in natural language processing (NLP). However, it is important to acknowledge the potential for implicit bias introduced by ChatGPT 4, which may stem from misinformation within the training data and a documented preference for text resembling the style of GPT-like models over other language model architectures, and sometimes even human-generated writing.

Below, we present the responses from both models to the first evaluation question from each category. It is worth noting that the questions were not sorted in any particular order, reflecting a random selection process.

- **When is it advisable to seek medical assistance for asthma?(factual)**
  - **Cellio**: Seeking medical assistance for asthma is advisable in several situations, including:
    1. _Severe Symptoms:_ If a person with asthma experiences severe symptoms such as difficulty breathing, chest tightness, wheezing, or coughing that does not improve with their usual medications.
    2. _Frequent Attacks:_ If the frequency of asthma attacks increases, it may indicate the need for a change in treatment or management plan.
    3. _Trouble Managing Symptoms:_ If someone is having difficulty managing their asthma symptoms despite following their treatment plan, it is important to seek medical advice.
    4. _Emergency Situations:_ In cases of an asthma emergency, where the person's breathing is severely compromised, medical assistance should be sought immediately.
  - **Gpt-4**: Based on the provided context from various studies and commentaries, seeking medical assistance for asthma should be considered in the following situations:
    1. _Deteriorating Symptoms:_ If an individual's asthma symptoms become more frequent or severe, or if they notice a significant decline in lung function as reflected in measurements like peak flow, it would be advisable to seek medical intervention.
    2. _Poorly Controlled Asthma:_ When one's asthma appears poorly controlled, with frequent need for quick-relief inhalers or disruption of daily activities, medical guidance can help adjust treatment plans for better control.
    3. _Exacerbations or Attacks:_ Seeking urgent medical help is critical during an asthma attack that doesn't improve with the use of short-acting bronchodilators. This is a potentially life-threatening situation.
    4. _Before a Clinical Trial:_ For individuals interested in participating in asthma or COPD clinical trials, consulting with a health professional is important to ensure their suitability and understand the implications of the study.
- **What are common triggers for asthma attacks?(List)**

  - **Cellio**: Common triggers for asthma attacks can vary depending on individual circumstances. Some of the common triggers include:
    1. Allergens such as pollen, dust mites, mold spores, pet dander, and cockroach droppings.
    2. Respiratory infections like colds, flu, and sinus infections.
    3. Irritants in the air such as cigarette smoke, air pollution, strong odors, and fumes.
    4. Physical activities, especially in cold or dry air.
    5. Weather changes, including cold air, humidity, or extreme temperatures.
    6. Strong emotions or stress.
  - **Gpt-4**: Asthma attacks can be triggered by a variety of factors, many of which are environmental or related to personal health habits and conditions. Common triggers for asthma attacks include:
    1. _Allergens:_ Such as pollen, animal dander, mold, and dust mites.
    2. _Respiratory infections:_ Like the common cold or the flu.
    3. _Physical activity:_ Exercise-induced asthma is a well-known phenomenon.
    4. _Cold air:_ Can lead to bronchial tube constriction.
    5. _Air pollutants:_ Including smoke, chemical fumes, and other irritants.
    6. _Stress and strong emotions:_ Which can lead to hyperventilation and an asthma episode.
    7. _Certain medications:_ For example, beta-blockers and aspirin can trigger asthma in some individuals.
    8. _Food additives:_ Some preservatives and sulfites in foods and beverages might be triggers.

- **What are the different eating disorder types?(list)**

  - **Cellio**: The different eating disorder types include Anorexia Nervosa (AN) and restrictive Anorexia Nervosa (AN-R). In relation to personality disturbances in AN, individuals with AN show increased scores on schizoid, borderline, avoidant, dependent, obsessive-compulsive, negativistic, and depressive personality dimensions compared to controls. These personality disturbances are related to negative mood states rather than specific eating disorder symptoms. Patients with AN-R exhibit significant impairment in information processing speed and planning, with two distinct subgroups identified based on cognitive functions and psychopathological symptoms. One subgroup shows more marked cognitive impairment and fewer psychopathological symptoms than the other subgroup and healthy control group.

  - **Gpt-4**:There are several well-characterized types of eating disorders, which are complex mental health conditions that can have serious physical and emotional consequences. The main types include:
    1. _Anorexia Nervosa (AN):_ Characterized by a pattern of restricting food intake which leads to significant weight loss or a lack of appropriate weight gain, intense fear of gaining weight, and a distorted body image. The context provided from the vector database discusses AN's association with personality disturbances, cognitive impairments, and potentially being an intermediate phenotype related to higher intelligence and working memory capacity.
    2. _Bulimia Nervosa (BN):_ Involves episodes of binge eating followed by compensatory behaviors like purging, fasting, or excessive exercise. People with BN may have a fear of weight gain, but unlike AN, they typically maintain a weight that is within or above the normal range. The context mentions that children of women with a lifetime diagnosis of BN showed comparatively poor visuo-spatial functioning, suggesting a potential cognitive phenotype associated with BN.
    3. _Binge Eating Disorder (BED):_ Similar to BN, BED includes episodes of binge eating but without the regular use of compensatory behaviors, leading to feelings of distress, guilt, and embarrassment. BED often leads to overweight or obesity.

- **Why is maintaining a healthy weight important? (Casual)**

  - **Cellio**: Maintaining a healthy weight is important for several reasons, especially in the context of the growing obesity epidemic in high-income countries. Precision nutrition, which focuses on tailoring dietary interventions to individuals based on their metabolic profile and environmental exposure, has emerged as a promising approach to address the complex and varied responses to diet that individuals may have. By maintaining a healthy weight, individuals can reduce their risk of developing chronic conditions like diabetes mellitus and obesity, which are associated with significant morbidity and mortality. Achieving an optimal weight can also lead to improved glycemic control and weight reduction, benefiting overall health outcomes. Precision nutrition offers personalized dietary recommendations that take into account individual variability in responding to nutrients, aiming to prevent and manage diseases effectively. With the aid of wearable and mobile sensors, timely and reliable predictions can be made for efficient feedback and guidance toward a managed nutritional balance. While large-scale interventions like precision nutrition can address individual causes of obesity, it is important to also consider the broader population causes embedded in social, cultural, economic, and political contexts that create obesogenic environments.

  - **Gpt-4**: Maintaining a healthy weight is important because it has a substantial impact on overall health and can help prevent a range of health concerns that are associated with being overweight or obese. From the context provided by the vector database, several perspectives on the importance of maintaining a healthy weight can be drawn:
    1. _Reducing the risk of obesity-related health conditions_: The papers highlight the prevalence of obesity and its association with socioeconomic status, indicating that groups of lower socioeconomic status are disproportionately affected. Obesity has been linked to numerous health issues, including cardiometabolic diseases, which are a major global health concern.
    2. _Personalized nutrition for effective interventions_: Research by Angeline Chatelan and others has emphasized the potential benefits of precision nutrition (PN), which provides personalized dietary advice compared to traditional 'one-size-fits-all' approaches. These personalized interventions, potentially enhanced by artificial intelligence and smartphone applications, could help individuals tailor their eating habits to better maintain a healthy weight and improve dietary intake.
    3. _Technological advancements in dietary behavior assessment_: Desiree A. Lucassen discusses the challenges in quantifying and influencing dietary behaviors, pointing out the limited success of conventional nutritional campaigns. With the Pride and Prejudice Consortium's focus on using AI, machine learning, and personalized coaching, the efforts to establish lasting dietary behavior change are more promising, which could lead to healthier weight management practices.
    4. _Real-world applications of artificial intelligence for nutrition_: Jingyuan Feng 2019s study on a novel PN approach supported by AI in a real-world scenario illustrates how technology can assist individuals with making informed food choices, leading to potential optimizations in dietary patterns, body weight, blood pressure, and general body metabolism.

- **Why is regular exercise beneficial for cardiac health? (Casual)**

  - **Cellio**: Regular exercise is beneficial for cardiac health for several important reasons. Firstly, regular exercise helps to improve cardiovascular health by strengthening the heart muscle and improving its efficiency. This can lead to lower resting heart rates, reduced blood pressure, and improved circulation throughout the body, ultimately reducing the risk of cardiovascular diseases such as heart attacks and strokes. Additionally, regular exercise can help to lower levels of bad cholesterol (LDL) and increase levels of good cholesterol (HDL), which can help to reduce the buildup of plaque in the arteries and decrease the risk of developing atherosclerosis. Moreover, regular exercise plays a crucial role in maintaining a healthy weight and reducing excess body fat, which can also contribute to a lower risk of heart disease and other related conditions like diabetes and high blood pressure. By engaging in regular physical activity, individuals can also improve their overall fitness levels, increase their stamina and endurance, and enhance their overall quality of life.
  - **Gpt-4**: Regular exercise is beneficial for cardiac health for several reasons, which can be understood by examining key findings from the research context provided:
    1. _Combating Sedentary Behavior:_ Prolonged sedentary behavior is positively associated with clustering of risk factors for cardiovascular disease (CVD). Regular physical activity counteracts the harmful effects of sedentary behavior by improving cardiovascular function and reducing risk factors such as obesity, high blood pressure, and glucose intolerance.
    2. _Personal Activity Intelligence (PAI):_ PAI is a metric that reflects the heart rate response to physical activity, considering age, sex, resting, and maximum heart rate. A score of PAI \u2265100 weekly has been shown to reduce the risk of premature CVD death in individuals, both healthy and those with known CVD risk factors. Regular exercise would help individuals achieve or maintain a PAI \u2265100, thereby offering protection against CVD mortality.
    3. _Temporal Changes in PAI and Mortality:_ Studies have shown that an increase in PAI over time, as well as maintaining a high PAI score, is associated with a reduced risk of CVD and all-cause mortality. This implies that not only is it important to engage in regular exercise but also to maintain or increase the intensity or frequency of exercise over time to ensure continued benefits for cardiac health.
    4. _Extension of Life Expectancy:_ Regular exercise and the associated improvement in PAI correlated with an increase in years of life gained. Participants who maintained or increased their PAI score over time saw substantial increases in life expectancy, further underscoring the benefits of regular exercise for cardiac health.

- **How would a person's health change upon quitting smoking?(hypothesis)**

  - **Cellio**: Based on the context provided, it is clear that chronic tobacco smoking can have negative effects on neuropsychological function. A systematic review and meta-analysis of 24 studies revealed a cross-sectional association between chronic tobacco smoking and impairments in cognitive impulsivity, attention, memory, intelligence, and cognitive flexibility. The study showed that chronic tobacco smokers performed worse on neuropsychological tests compared to non-smokers in these domains. The largest effect size was related to cognitive impulsivity, while the smallest effect size was related to intelligence.In contrast, studies on cannabis use and neuropsychological performance have shown mixed results. While there is some evidence of a low association between chronic cannabis use and impairments in cognitive impulsivity, attention, memory, and flexibility, the findings are inconclusive.
  - **Gpt-4**: Upon examining the context provided from the vector database and considering the relevant literature, quitting smoking would have several beneficial effects on a person's health, particularly in terms of neurocognitive functioning. Based on the systematic review and meta-analysis by A. A. Conti (Reference ID: 30502351), chronic tobacco smoking is associated with neuropsychological impairments across several domains including cognitive impulsivity, attention, intelligence, short term memory, long term memory, and cognitive flexibility. However, the study highlighted no significant association with motor impulsivity.\n\nIn the context of Durazzo's research (Reference IDs: 23682867 presented thrice, which may indicate an input duplication error), the focus was on individuals with alcohol dependence and their neurocognition in relation to chronic cigarette smoking. According to the study, actively smoking alcohol-dependent individuals (asALC) were compared with never-smoking alcohol-dependent individuals (nvsALC) and former-smoking alcohol-dependent individuals (fsALC) on a comprehensive neurocognitive battery. They found that actively smoking alcohol-dependent individuals displayed steeper age-related effects on various neurocognitive domains compared to their non-smoking counterparts. It's important to note that while this study focused on alcohol-dependent individuals, the implications it raises about the interactive effects of smoking and neurocognition are still relevant. Here are some changes that can typically be expected upon quitting smoking, inferred from the available context and wider health literature:
    1. Neurocognitive improvements: As inferred from the studies mentioned, chronic smoking is associated with impairments in several neurocognitive functions, and quitting smoking may help in recovering these functions over time.

- **How do sedentary living and a high-sugar diet contribute to type 2 diabetes development? (Complex)**

  - **Cellio**: Sedentary living and a high-sugar diet can contribute to the development of type 2 diabetes through multiple mechanisms.

    1. \*Insulin Resistance: A sedentary lifestyle can lead to insulin resistance, where the body's cells become less responsive to insulin. This means that glucose is not effectively taken up by cells, leading to elevated blood sugar levels over time. Similarly, a high-sugar diet can cause persistent high levels of glucose in the blood, further contributing to insulin resistance.
    2. _Weight Gain_: Sedentary living and a diet high in sugar can also contribute to weight gain and obesity. Excess body weight is closely linked to the development of type 2 diabetes as it further exacerbates insulin resistance and leads to chronic inflammation, both key components in diabetes development.
    3. _Inflammation_: Both sedentary living and a high-sugar diet can promote inflammation in the body. Chronic inflammation is associated with an increased risk of developing type 2 diabetes as it can disrupt insulin signaling and overall metabolic function.

  - **Gpt-4**: Considering the provided context, the contribution of sedentary living and a high-sugar diet to the development of type 2 diabetes can be discussed from several perspectives:
    1. _Deterioration of Glucose Tolerance with Age_: With age, there\u2019s a natural decline in glucose tolerance. However, in individuals leading a sedentary lifestyle and consuming a high-sugar diet, this deterioration can be exacerbated. These behaviors can speed up the decline in the body's ability to regulate blood sugar levels, thereby increasing the risk for type 2 diabetes.
    2. _Glucose Trajectory and Mathematical Modeling_: Mathematical models, like those mentioned in the context from Swislocki (35426740), analyze longitudinal changes in metabolic markers such as fasting plasma glucose and glycated hemoglobin (HbA1c). Sedentary living and high-sugar diets can influence these trajectories, leading to a higher likelihood of developing a trajectory that is associated with the onset of diabetes.
    3. _Genetic Contributions and Lifestyle Factors_: Although genetic predisposition plays a role, lifestyle factors such as diet and physical activity level are critical. An unhealthy diet high in sugar and lack of physical activity can contribute to weight gain and insulin resistance, both of which are significant risk factors for type 2 diabetes, even for those with a genetic predisposition.

The experiment was conducted on a Windows 11 desktop using a intel core i7 with 16GB of DDR4 DRAM and an NVIDIA GeForce RTX 3050 4GB.

## Conclusion

This report introduces Cellio, A complex RAG system using the turbo GPT 3.5 model. Our system make use of query transformation to fine tune queries for more accurate data retrieval from a vector store. Retrieved documents are more refined via context concatenation, and semantic filtering. Turbo-GPT 3.5 performs similarly to `gpt-4-1106-preview` given the same context and reasonably accurate results. As is shown in this project report, Cellio is effective as a QA system for answering questions about PubMed abstracts and related topics.

## References

<a id="1">[1]</a>GPTs

    A.Vaswani, N.Shazeer, N.Parmar, J.Uszkoreit, L.Jones, A.N.Gomez, L.Kaiser, I.Polosukhin, Attention is All You Need,  in: Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, Curran Associates Inc., RedHook, NY, USA, 2017, p.6000–6010.

<a id="2">[2]</a> Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

    Patrick Lewis and Ethan Perez and Aleksandra Piktus and Fabio Petroni and Vladimir Karpukhin and Naman Goyal and Heinrich Küttler and Mike Lewis and Wen-tau Yih and Tim Rocktäschel and Sebastian Riedel and Douwe Kiela,
    2021,

<a id="3">[3]</a> RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation

    Fangyuan Xu and Weijia Shi and Eunsol Choi,
    2023,
    https://arxiv.org/abs/2310.04408

## Useful Tutorials

1. https://medium.com/@jerome.o.diaz/langchain-conversational-retrieval-chain-how-does-it-work-bb2d71cbb665
   Helped in understanding how to work with langchain for retrieval

2. https://python.langchain.com/docs/integrations/llms/openlm
   Helped in understanding the workings of OpenAI Language models api

3. https://community.openai.com/t/providing-context-to-the-chat-api-before-a-conversation/195853
   Helped in understanding the chat expect of using models and how to use context for querying the models
4. https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
   Helped in understanding the concepts of multi-query retrieval and its implementations
