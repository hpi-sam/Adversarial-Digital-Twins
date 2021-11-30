package de.mdelab.morisia.selfhealing;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;

public class ArchitectureUtilCal {
	public static double computeOverallUtility(Architecture architecture) {
		double overallUtility = 0;
		for (Tenant shop : architecture.getTenants()) {
			overallUtility += computeShopUtility( shop) ;
		}
		//System.out.println("\n Architecture u: " + overallUtility);
		return overallUtility;
	}

	
	



	public static double computeShopUtility(Tenant shop) {
		double shopUtility = 0;
		for (Component component : shop.getComponents()) {
			shopUtility+=computeComponentUtility(component);
		}
		//System.out.println("\n shop utility: " + shopUtility);
		return shopUtility;
	}



	public static double computeComponentUtility(Component component) {
		double componentUtility = 0;
		double reliability = component.getType().getReliability();
		double outgoingConnectors = component.getRequiredInterfaces().size();
		double incomingConnectors = 0;
		double connectivity=incomingConnectors+outgoingConnectors;
		
		//calculate combined utility
		
		int numberOfConnectors = component.getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : component.getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}
		long start_P = System.nanoTime();

		componentUtility= component.getType().getReliability() * component.getType().getCriticality()
				* component.getPerformance()
				* component.getTenant().getImportance()
				* (component.getRequiredInterfaces().size() + 1);

		int numberOfProvided = 0;
		for (ProvidedInterface pi : component.getProvidedInterfaces()) {
			numberOfProvided = numberOfProvided + pi.getConnectors().size();
		}

		if (numberOfProvided > 1) {
			componentUtility = componentUtility * numberOfProvided
					- 10 * component.getADT();
		}

		else if (numberOfProvided == 1) {
			componentUtility = componentUtility * numberOfProvided * 2
					- 10 * component.getADT();
		}

		else {
			componentUtility = componentUtility- 10 * component.getADT();
		} 
		
	//	System.out.println("\n Component utility: " + componentUtility);
		componentUtility=componentUtility-computeUtilityDrop(component);
		
		//System.out.printf("\n -----Component Utility of "+component.getType().getName()+"  is  "+componentUtility);
		
		{
			
		}
		
		return componentUtility;
	}
	public static double computeUtilityDrop(Component component) {
		double utilityDrop = 0;
		
		
	
		for (Issue issue : component.getIssues()) {
			utilityDrop+=issue.getUtilityDrop();
			
		}
		
			//System.out.println("\n\n !! Drop in Component " +component.getType().getName()+"Detected: " + utilityDrop);
		return utilityDrop;
	}

	
}
