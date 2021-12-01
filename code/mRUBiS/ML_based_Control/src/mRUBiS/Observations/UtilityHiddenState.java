package mRUBiS.Observations;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.selfhealing.ArchitectureUtilCal;

import java.util.Random;

/**
 * Keeps the hidden utility states of each component type at a given shop.
 * The states are hidden because they are produce by a time series generator (auto-regressive model).
 * This generative process is invisible to the controller or any external agent.
 *  
 * @author Christian Adriano
 *
 */
public class UtilityHiddenState {

	/** key,value pairs in which the key can be a string that concatenates shop and component_type*/
	public HashMap<String,Double> CurrentUtilityStateMap = new HashMap<String,Double>();

	/** holds the utility values that each component should converge to */
	public HashMap<String,Double> ReferenceUtilityStateMap = new HashMap<String,Double>(); 

	/** convergence rate towards the utility of reference */
	//private Double theta = 0.01;
	private Double theta = 0.05;

	/** rate to compute the variance around the current utility */
	private Double sigma = 1.0;

	/** object to generate Gaussian numbers, initial seed=9876 */
	private Random randomGenerator = new Random(9876);
	
	 /** determines how far from the "referenceUtility" should we start. 
	  * The default is 1.5, which implies a 150% of the reference utility value*/
	private Double delta = 1.5; 
	
	/**
	 * Constructor that uses the default parameters values
	 */
	public UtilityHiddenState(Architecture mRUBiS) {
		this.intilializeReferenceUtilityMap(mRUBiS);
		this.initializeCurrentUtilityMap();
	}
	
	/** 
	 * In case one ones to change the standard calibration of the auto-regressive model
	 * @param theta convergence rate towards the utility of reference
	 * @param sigma rate to compute the variance around the current utility
	 * @param randomSeed initial seed to add noise to the auto-regressive model that updates the utility of each component
	 * @param delta determines how far from the "referenceUtility" should we start
	 */
	public UtilityHiddenState(Architecture mRUBiS,Double theta, Double sigma, long randomSeed, Double delta) {
		this.theta =  theta;
		this.sigma = sigma;
		this.randomGenerator =  new Random(randomSeed); 
		this.delta = delta;
		this.intilializeReferenceUtilityMap(mRUBiS);
		this.initializeCurrentUtilityMap();
	}
	

	/**
	 * For all components of the mRubis instance, 
	 * obtain their utility values and store under a key "shop:componentType"
	 */
	private void intilializeReferenceUtilityMap(Architecture mRUBiS) {
		
		for(Tenant shop:mRUBiS.getTenants())
		{	String shopName=shop.getName();
		for (Component component:shop.getComponents())
		{String componentType=component.getType().getName();
		if (componentType.contains("Authentication Service"))
		{componentType="Authentication Service";}
		double referenceUtility = ArchitectureUtilCal.computeComponentUtility(component);
		String key =shopName+":"+componentType;
		this.ReferenceUtilityStateMap.put(key, referenceUtility);
		 System.out.println("\n For Component "+ componentType+ " in shop "+ shopName+ " Reference Utility is "+referenceUtility);
		}

		}
		}


	
	/**
	 * For each component in the ReferenceUtilityStateMap:
	 * currentUtility = referenceUtility * this.delta
	 */
	private void initializeCurrentUtilityMap() {
		// Getting an iterator
        Iterator<Entry<String, Double>> hmIterator = this.ReferenceUtilityStateMap.entrySet().iterator();
  
        // Iterate through the hashmap
        // and add some bonus marks for every student
      //  System.out.println("HashMap after adding bonus marks:");
  
        while (hmIterator.hasNext()) {
            Map.Entry<String,Double> mapElement = (Map.Entry<String,Double>)hmIterator.next();
            String key = ((String)mapElement.getKey());
            Double referenceUtility = ((Double)mapElement.getValue());
            Double currentUtility = referenceUtility * this.delta;
            this.CurrentUtilityStateMap.put(key,currentUtility);
            System.out.println("\n  --- Current Utility for key "+ key+" is "+currentUtility); }
	}

	/**
	 * Implements the auto-regressive model combined with an Ornstein�Uhlenbeck procedure.
	 * @param shopName the key name of a shop
	 * @param componetType the key name of a component type
	 * @return currentUtility which is a utility shifted closer to the referenceUtility
	 */
	public  Double getCurrentUtility(Component component) {

		String componentType=component.getType().getName();
		if (componentType.contains("Authentication Service"))
		{componentType="Authentication Service";}
		String shopName=component.getTenant().getName();
		String key = shopName+":"+componentType;


		if(!this.CurrentUtilityStateMap.containsKey(key)) {
			//Invalid Operation
			System.out.println("\n key " + key+ " is Invalid!!!!");
			return(null);
		}
		else {//Compute the new utility based on previous one
			Double previousUtility = (Double) this.CurrentUtilityStateMap.get(key);
			Double referenceUtility = (Double) this.ReferenceUtilityStateMap.get(key);
			double variance = this.sigma.doubleValue() * this.randomGenerator.nextGaussian(); //nextGaussian samples from a normal distribution with mean=0,std=1
			double convergenceShift = this.theta.doubleValue() * (referenceUtility.doubleValue() - previousUtility.doubleValue());
			double currentUtility = previousUtility.doubleValue() +  convergenceShift + variance;
			
			//Stores new state
			this.CurrentUtilityStateMap.put(key,currentUtility);
			 System.out.println("\n U is " + currentUtility);
			  
			return currentUtility;
		}
	}
	
	/**
	 * This method allows to reset the current utility of particular component. 
	 * This should happen when a component has been restarted.
	 * @param shopName the key name of a shop
	 * @param componetType the key name of a component type
	 */
	public void resetUtilityState(String shopName, String componentType) {
		if (componentType.contains("Authentication Service"))
		{componentType="Authentication Service";}
		String key = shopName+":"+componentType;
		Double referenceUtility = (Double) this.ReferenceUtilityStateMap.get(key);
		this.CurrentUtilityStateMap.put(key,referenceUtility*this.delta);		
	}


}
