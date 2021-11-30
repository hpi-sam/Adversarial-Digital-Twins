package de.mdelab.morisia.selfhealing.rules;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.CF1;
import de.mdelab.morisia.comparch.CF2;
import de.mdelab.morisia.comparch.CF3;
import de.mdelab.morisia.comparch.CF5;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.HwRedeployComponent;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.LwRedeployComponent;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.RestartComponent;
import de.mdelab.morisia.comparch.AddReplica;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.selfhealing.Approaches;
import de.mdelab.morisia.selfhealing.Evaluation_ML;


public class PerformanceEfficiencyManager {

	private static Random random = new Random();

	private final static Logger LOGGER = Logger.getLogger(PerformanceEfficiencyManager.class
			.getName());

	/**
	 * For the given component, the method calculates the performance
	 * 
	 * @param component
	 *            the given component.
	 * @return 
	 * @return 
	 */
	public static double calculatePerformance(Component component) {
		
	double pMax=component.getType().getPerformanceMax();
	double req=component.getRequest();
	double rep=component.getInUseReplica();
	double alpha=4/component.getType().getSatPoint();
	double performance=pMax*Math.tanh(alpha*rep/req);

	component.setPerformanceOld( component.getPerformance());
	double delta = performance-component.getPerformanceOld();
	/*System.out.print("\n\n Component "+
	component.getType().getName()
	+"\n Insue   "
			+rep+
			"\n PMax "+ pMax+ ",  New Perfprmance " + performance
			+ " old P"+ component.getPerformanceOld()
			+ "\n delta  "+ delta);*/
	return (performance);
	}
	
	/**
	 * For the given component, the method calculates the efficiency
	 * 
	 * @param component
	 *            the given component.
	 */
	public static double calculateEfficiency(Component component) {

		
		double pMax=component.getType().getPerformanceMax();
		double req=component.getRequest();
		double rep=component.getInUseReplica();
		double alpha=4/component.getType().getSatPoint();
		double performance=pMax*Math.tanh(alpha*rep/req);
		double efficiency=performance
			/(component.getInUseReplica()
			*component.getType().getReplicaADT()
			*component.getType().getReplicaCost());
	
		return(efficiency);
	}

	

	
}//EOC
