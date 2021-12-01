package de.mdelab.morisia.selfhealing;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class LRIBasedScenario {

	public static void main(String[] args) throws IOException {
		
		
		// LogN parameters for Group size
		Random rngSize = new Random(0);
		Random rngIat = new Random(0);
		int length = 1500;
		double[] lognormalValuesSize = new double[length];
		double[] lognormalValuesIat = new double[length];
		
		
		int[] groupSize = new int[length];
double stddevIat= 1.28;
double meanIat = -1.46;

double stddevSize= 0.77;
double meanSize = 1.32;
FileWriter fileWriter = null;
 fileWriter = new FileWriter("measurements/TAAS18/LRI05_Injection_Statistics.csv");
fileWriter.append("Group Size" + "," + "IAT (sec)" +"\n");



		for (int i = 0; i < length; ++i)
		{
			
			
			double stdNormalSize = rngSize.nextGaussian();
			double stdNormalIat = rngIat.nextGaussian();
			
			double normalValueSize = stddevSize*stdNormalSize + meanSize;
			double normalValueIat = stddevIat*stdNormalIat+ meanIat;
			
		    
			lognormalValuesSize[i] = Math.exp(normalValueSize);
			groupSize[i]= (int) Math.ceil(lognormalValuesSize[i]);
			lognormalValuesIat[i] = Math.floor( Math.exp(normalValueIat)*3600);
			fileWriter.append(groupSize[i] + "," + lognormalValuesIat[i] +"\n");
			
			//System.out.println(lognormalValues[i]+"       ");
		//	System.out.println(groupSize[i]+"\n");
		}
		 
		fileWriter.flush();
		fileWriter.close();
		
	}

}
