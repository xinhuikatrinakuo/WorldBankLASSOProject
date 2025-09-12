<h1>Predicting Household Poverty Status Using Rapid Survey Data</h1>

<h2>Description</h2>
Identifying households in poverty is essential for providing effective assistance, but traditional surveys are expensive and time-consuming. This project uses World Bank survey data to build machine learning models that predict poverty using a small set of encoded variables. The aim is to find the most important predictors to make poverty identification faster and more cost-effective. In order to evaluate performance, the models are assessed using the logloss metric, which rewards confident correct predictions while penalizing overconfident mistakes. Overall, the project includes data analysis, careful model selection and tuning, identification of key features, and well-documented code to support reproducibility.
<br />


<h2>Data</h2>

Analysis leverages two datasets: a training set and a test set, each consisting of household survey responses. Each observation corresponds to a unique household and is labeled with a binary poverty indicator (`Poor`). 
<p align="center">
<img width="457" height="258" alt="Screenshot 2025-09-12 at 3 46 39 PM" src="https://github.com/user-attachments/assets/1817f46f-9414-45bb-b26e-f1ba15e5a2f4" />
<br>
<p align="left">
Survey variables include both categorical indicators and numeric measures, but all are encoded as random character strings. For example, categorical variables may reflect household ownership of items, while numeric variables could represent quantities like the number of working cell phones or the number of rooms in the household. Because the variables are encoded, the focus of this project is on selecting the most predictive features rather than interpreting their real-world meaning.

<h2>Methodology</h2>

Project applies machine learning classification techniques to predict household poverty status, with performance evaluated using the logloss error metric. Logloss rewards confident, correct predictions and penalizes overconfident misclassifications, providing a nuanced measure of probabilistic model performance. Models are selected, tuned, and evaluated to identify key variables and achieve optimal predictive accuracy.


<h2>Program walk-through:</h2>

<p align="center">
Launch the utility: <br/>
<img src="https://i.imgur.com/62TgaWL.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Select the disk:  <br/>
<img src="https://i.imgur.com/tcTyMUE.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Enter the number of passes: <br/>
<img src="https://i.imgur.com/nCIbXbg.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Confirm your selection:  <br/>
<img src="https://i.imgur.com/cdFHBiU.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Wait for process to complete (may take some time):  <br/>
<img src="https://i.imgur.com/JL945Ga.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Sanitization complete:  <br/>
<img src="https://i.imgur.com/K71yaM2.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<br />
Observe the wiped disk:  <br/>
<img src="https://i.imgur.com/AeZkvFQ.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
</p>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
