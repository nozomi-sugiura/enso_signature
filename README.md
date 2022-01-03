# enso_signature
You can predict El Nino using climate indices from NOAA.
After downloading this directory, all you do is as follows.
 tar xvzf data.tgz;
 ./enso_signature.py;
 gnuplot pred.gp
The predicted values are written as  separate file for each month. 
Required Python libraries are sys, numpy, scikit-learn, esig.
For control experiment, run ./enso_signature_cnt.py.

