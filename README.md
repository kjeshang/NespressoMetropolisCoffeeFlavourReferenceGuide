# NespressoMetropolisCoffeeFlavourReferenceGuide
> **Kunal Jeshang - Team Leader** *(Project Timeline: December 2023 -- January 2024)*

## Table of Contents

## Introduction
This project was inspired by the [Nespresso Metropolis Training App](https://github.com/kjeshang/NespressoMetropolisTrainingApp) project, as well as the day-to-day aspects of providing recommendations to customers regarding our coffee lineup. The drawback of the aforementioned project was that it was far too gradiose and could not be put into production/functional use for the business operations. Furthermore, as it was a custom made web application, the challenge in adopting it would be receiving corporate level approval to perform technology stack setup and server deployment costs. A decisive reason why I decided to embark on this project was due to my promotion from Coffee Specialist to Team Leader, which at times requires me to become an adept knowledge expert for both the customer and the other coffee specialists. As a newly-promoted & inexperienced Team Leader during a busy season, it was important to "hit-the-ground running" and make a positive impact to support the team. Thus, I wanted to create a coffee flavour reference guide that I can easily access on my mobile phone to provide answers to questions that I may not know already. When conceptualizing the project, I initally thought that I could code an Android mobile application with CRUD functionality, akin to the [Staples Marine Way Associate Companion App](https://github.com/kjeshang/StaplesMarineWayAssociateCompanionMobileApp). However with Nespresso having a smaller-concise product lineup, I felt that this would not be necessary and coding an Android application would require a long development time. Therefore, I decided to create static reference guides for each coffee flavour on the Nespresso menu, which are saved in a folder directory app on my mobile phone and can be retrieved quickly using the built-in search bar feature (i.e., keyword/document name search). Using Python programming, the reference guides would be procedurally generated as PDF documents with an Excel workbook as the data source. A reference guide for a given coffee would contain general information, taste profile description & metrics, and feature results & recommendations.

## Technology Requirements
* Computer
    * ASUS VivoBook Windows 11
* Programming Language
    *  Python (Anaconda Distribution)
    *  _Important Packages:_ Pandas, Numpy, Matplotlib, Seaborn, WordCloud, Requestsm, PIL, shutil, urlib, io, imageio, pillow_avif, NLTK, string, re, Scikit Learn, ReportLab 
* Software
    * Microsoft Visual Studio Code
    * Google Sheets/LibreOffice Calc
    * Web Browser (e.g., Google Chrome)
    * Anaconda Navigator
    * Jupyter Notebook

## Data Collection
The textual data was collected manually from the official Nestle Nespresso Canada website. The capsule images were also retrieved from the official website as well. The sleeve images were retrieved from various website with the help of Google images. All of the aforementioned data was populated in an Excel Workbook. The actual capsule & sleeve images were also downloaded and saved in a raw images directory for future local retrieval. Below is a tabular breakdown of the raw dataset that would later be used in the forthcoming analysis, which encapsulates image conversions, data cleaning, chart creation, machine learning, and report generation.

|Column Name|Data Type|Constraint|Notes|
|--|--|--|--|
|ID|Text|Not Null|Unique identifier|
|Name|Text|Not Null|Name of coffee|
|Capsule Image|Text|Not Null|Hyperlink of capsule image|
|Sleeve Image|Text|Not Null|Hyperlink of sleeve, capsule, and cup image|
|Type|Text|Not Null|Vertuo or Original line capsule|
|Status|Text|Not Null|Current, Not Current, Limited Edition, Seasonal|
|Category|Text|Not Null|Coffee flavour or size category on the menu|
|Cup Size|Text|Not Null|Metric volume of the coffee in milli-liters, as well as black coffee/milk recipe type calssification|
|Headline|Text|Not Null|Defining trait of the coffee|
|Intensity|Numerical|Null|Overall strength level of coffee|
|Price|Numerical|Not Null|Sleeve price of coffee (typically 10 capsules, and 7 capsules for Vertuo Alto & Carafe)|
|Notes|Text|Null|Aromatic notes/smell of the coffee|
|Taste|Text|Not Null|Taste of the coffee|
|Acidity|Numerical|Null||
|Bitterness|Numerical|Null||
|Roastiness|Numerical|Null||
|Body|Numerical|Null||
|Milky Taste|Numerical|Null||
|Bitterness with Milk|Numerical|Null||
|Roastiness with Milk|Numerical|Null||
|Creamy Texture|Numerical|Null||
|Description|Text|Null|The story and general explanation about the coffee|
|Origin|Text|Null|Origin of coffee bean|
|Roasting|Text|Null|Roasting process to create coffee|
|Contents & Allergens|Text|Not Null||
|Ingredients|Text|Not Null||
|Net Weight|Text|Not Null||

## Analysis Process Flow
1. Import Dataset
    * Import necessary Python packages.
    * Import raw dataset, containing coffee information and hyperlinks to images, as a Pandas dataframe.
    * Take a peak at the newly imported data.
2. Create a duplicate of the raw dataset so that further analysis can be conducted without disrupting the integrity of the imported raw dataset.
3. Data Cleaning, Normalization, and Aggregation
    * Create unique names for some flavours that share the same name.
    * Estimate intensity level for some flavours as some are do not have a prescribed level by Nespresso (e.g., Barista Creation, Vertuo Carafe, etc).
    * Determine intensity classification based on actual/estimated intensity.
        |Intensity Range|Classification|
        |--|--|
        |0 < Intensity < 5|Blonde|
        |5 <= Intensity <= 7|Medium|
        |7 < Intensity <= 10|Dark|
    * Estimate taste profile classification (i.e., Acidity, Bitterness, Roastiness, Body, Milky Taste, Bitterness with Milk, Roastiness with Milk, Creamy Texture).
        |Taste Profile Range|Classification|
        |--|--|
        |0 < Taste Profile Level <= 2|Low|
        |2 < Taste Profile Level < 4|Medium|
        |4 <= Taste Profile Level <= 5|High|
    * Create new columns for **Unique Name**, **Intensity Classification**, and every type of **Taste Profile Classification**.
4. Convert Raw Images
    * As mentioned in the "Data Collection" section of the report, the images were already manually downloaded and saved in a Raw Images directory. Most of the raw images are in _WEBP_, _AVIF_, or _JPEG_ format. It is now time to convert the images to PNG format. 
    * After the images are converted they are saved in the Images directory.
    * The **Capsule Image** and **Sleeve Image** column values are updated to reflect the local relative filepaths of the newly converted images that are in PNG format.
5. Create Taste Profile Charts
    * Create taste profile level charts for all coffee flavours and save it as a PNG image in the Charts directory.
    * In the event that a coffee flavour does not have a prescribed taste profile level (i.e., Ice Coffee), a taste profile chart will not be created.
    * Create a new column called **Taste Profile Chart** and populate the column values with the local relative filepaths of taste profile PNG images.
7. Perform Textual Pre-Processing
    * Take the most important and informative columns and combine them into a singular text value. The ID, Name, Capsule Image, and Sleeve Image columns would be excluded as it would construe the results when performing future machine learning.
    * The following steps are performed on the combined-singular "Textual Info" value.
        * Tokenization: Split all words of the chunked textual features by a space, and add all words as elements/tokens to a list.
        * Lemmatization: Reduce extended words (i.e., tokens) into their base word (e.g., Convert "Affected" to "Affect").
        * Remove instances of Stop Words, Punctuation, and Numerical values, which may construe future machine learning results. 
        * Part-of-Speech (POS) Tagging: Identify parts of speech of the words, and filter out the words that are of the following POS tag and grammatical classification.
        * 'Chunk' together all pre-processed textual features together.
    * Create a new column called **Textual Info** and populate the column values with the chunked textual features.
7. Check the final dataset and acknowledge the newly created columns, and the columns where the values were changed compared to that of the raw dataset.
    |Column Name|Data Type|Constraint|Notes|
    |--|--|--|--|
    |Capsule Image|Text|Not Null|Local relative filepath of capsule image|
    |Sleeve Image|Text|Not Null|Local relative filepath of sleeve image|
    |Unique Name|Text|Not Null|Name of Coffee that is unique and does not conflict with any other coffee|
    |Estimated Intensity|Text|Not Null||
    |Intensity Classification|Text|Not Null|Blonde, Medium, Dark|
    |Acidity Classification|Text|Not Null|Low, Medium, High|
    |Bitterness Classification|Text|Not Null|Low, Medium, High|
    |Roastiness Classification|Text|Not Null|Low, Medium, High|
    |Body Classification|Text|Not Null|Low, Medium, High|
    |Milky Taste Classification|Text|Not Null|Low, Medium, High|
    |Bitterness with Milk Classification|Text|Not Null|Low, Medium, High|
    |Creamy Texture Classification|Text|Not Null|Low, Medium, High|
    |Textual Info|Text|Not Null|Pre-processed textual features|
8. 









