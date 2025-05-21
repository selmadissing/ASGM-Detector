<p align="center">
  <h1 align="center">Thesis Project Logbook</h1>
</p>

Here, I keep track of my weekly progress, outlining what I’ve worked on, what’s coming next, and any thoughts along the way.  
The goal is to stay organized, reflect regularly, and make it easier to share updates with my supervisor.

---


## Week of 31 March – 4 April 2025

### Main Activities and Accomplishments
- Organize myself and understand what has been done / what needs to be done.
- Shortened the introduction based on feedback
- Started on the actual thesis template in Overleaf

### Plans for Next Week
- Methodology sections
- Description of data and study region
- EDA
- Start on the integration of satellite patches and OSM data

### Challenges, Questions, or Notes
- This week was mostly focused on remininding myself what I have done so far in terms of the project and understand what the next steps are.
---


## Week of 7 April – 11 April 2025

### Main Activities and Accomplishments
- Write on the Methodology section with focus on data and study region, EDA and creating patches for OSM
- Decided which OSM features to use
- Understand the pipeline from the baseline mining detector better

### Plans for Next Week
- How to overlay the patches
- Methodology pipeline diagram
- Finish Methodology for milestone deadline

### Challenges, Questions, or Notes
- Challenge: how to do the overlay, should I keep different colours for each OSM feature? I think so. Try out both.
- Question: What to do about the class imbalance? How is it dealt with?
- Question: Are more plots necessary for EDA? The reflectance bands pixel range?
- Question: Where do I show the locations of the sampling points? Where do I discuss it? Maybe EDA? 
---


## Week of 14 April – 18 April 2025

### Main Activities and Accomplishments
- Developed and refined the methodology for the upcoming milestone.
- Successfully retrieved the correct OSM tiles for the study region.
- Explored different strategies for incorporating OSM data: overlaying masks, adding as extra channels, or integrating as a separate input branch in the CNN architecture.

### Plans for Next Week
- Implement the approach of integrating OSM data as additional channels to the Sentinel-2 input.

### Challenges, Questions, or Notes
- Experienced uncertainty regarding the methodology and changed direction several times.
- Was sick for 2 days
---


## Week of 21 April – 25 April 2025

### Main Activities and Accomplishments
- Successfully generated image patches with OSM data included as additional channels.

### Plans for Next Week
- Finalize the experimental setup and begin training the models.


### Challenges, Questions, or Notes
- Needed to determine the correct format for saving the stacked image patches to ensure compatibility with the model pipeline.
---


## Week of 28 April – 2 May 2025

### Main Activities and Accomplishments
- Set up environment and jobs on Snellius for model training.
- Finalized experimental setup.
- Reached out to Earth Genome to inquire about their methodology for sampling negative points.

### Plans for Next Week
- Train all model variants using the prepared datasets.

### Challenges, Questions, or Notes
- Need to decide whether to use Earth Genome’s handpicked negative points or generate a new set of random negative samples.
- Working on finalizing the Snellius job script for large-scale training.
---


## Week of 5 May – 9 May 2025

### Main Activities and Accomplishments
- Generated random negative sampling points; used a subsample due to local storage constraints.
- Successfully trained initial models both with and without OSM features.

### Plans for Next Week
- Run experiments varying: patch sizes, class balance weights, negative sample strategies (curated vs. random)

### Challenges, Questions, or Notes
- Initially planned to use all non-mining points as negatives, but this proved infeasible due to storage limitations—unable to download all corresponding image patches locally.
---


## Week of 12 May – 16 May 2025

### Main Activities and Accomplishments
- Conducted experiments varying patch sizes, class balance weights, and negative sampling strategies (curated vs. random).

### Plans for Next Week
- Analyze experimental results and assess performance trends across configurations.

### Challenges, Questions, or Notes
- Initial results were unexpected; upon review, I identified an error in the data splitting strategy.
- As a result, all experiments had to be rerun to ensure consistency and validity.
- Was on a short holiday break for part of the week
---


