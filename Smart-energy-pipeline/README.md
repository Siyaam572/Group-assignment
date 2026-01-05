\# Smart Energy Grid Optimization Pipeline



Machine learning pipeline for optimizing power plant selection for the National Energy Consortium (NEC).



\## Setup



\### Installation



1\. Install dependencies:

```bash

pip install -r requirements.txt

```



2\. Place data files in the `data/` folder:

&nbsp;  - demand.csv

&nbsp;  - plants.csv

&nbsp;  - generation\_costs.csv



\## Usage



Run the entire pipeline:

```bash

python main.py

```



\## Changing Models



Edit `config/config.yaml` and change:

```yaml

model:

&nbsp; type: "RandomForest"  # or "GradientBoosting"

```



\## Team



\- Pipeline Architect: \[Your Name]

\- Data \& Evaluation Engineer: \[Person B]

\- Model Specialist: \[Person C]

\- Documentation Lead: \[Person D]

\- Visuals \& QC Lead: \[Person E]

