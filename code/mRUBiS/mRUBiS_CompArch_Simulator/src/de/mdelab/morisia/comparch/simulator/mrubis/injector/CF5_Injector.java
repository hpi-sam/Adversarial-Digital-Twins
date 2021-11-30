package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.Random;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;

/**
 * Injects a {@link IssueType#CF5}.
 * 
 * @author sona ghahremani
 * @version 0.21
 *
 */
public class CF5_Injector implements Injector<Component> {

	private final static Logger LOGGER = Logger.getLogger(CF5_Injector.class
			.getName());

	/**
	 * Constructor.
	 */
	public CF5_Injector() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "CF5 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Drops the efficinecy of a STARTED component.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.CF5;
	}

	/**
	 * Injects a CF5 into the given component.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Component component) {
		if (component.getState() == ComponentLifeCycle.STARTED) {
			 //Request per component
			int maxReq=500;
			int minReq=50;
			Random rn = new Random();
			double oldrequest=component.getRequest();
			double request = rn.nextInt((maxReq - minReq) + 1) + minReq;
			//double request = 4586;
			component.setRequest(request);
			
			//update Performance with regard to new request
			double oldperformance=component.getPerformance();
			double Newperformance=component.getType().getPerformanceMax()
					        *Math.tanh((4/component.getType().getSatPoint())
							*(component.getInUseReplica()/request));
			System.out.print(" \n\n Old P   "+ oldperformance+ "    Old Request   "+oldrequest
					+ "\n New P  "+ Newperformance +"   New Request  "+ request );
			component.setPerformance(Newperformance);
			// calculate the caused drop:
			int numberOfConnectors = component.getRequiredInterfaces().size();
			// second, add number of incoming connectors
			for (ProvidedInterface pi :component.getProvidedInterfaces()) {
				numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
			}
			double drop=component.getType().getReliability()
					*component.getType().getCriticality()
					*numberOfConnectors
					*Newperformance;
			//System.out.print(" \n\n  5: drop: "+ drop);
		
	     //   System.out.print("\n\n\n\n in Use"+component.getInUseReplica() );
	    //    System.out.print("\n should be in use "+(Math.ceil(request*component.getType().getSatPoint())) );
			LOGGER.info(" -- Change the request of component\""
					+ component.getUid() + "\" from tenant \""+ component.getTenant().getName());
			
		/*	System.out.print(" \n-- Change the request of component\""
					+ component.getType().getName()+ "\" from tenant \""+ component.getTenant().getName() + "  From:  " + oldrequest+"  To:  "+ request
					+" \n P       : "+Newperformance+"\n Max/old P " + oldperformance+ " max   "+ component.getType().getPerformanceMax()
					);*/

			return true;
		} else {
			LOGGER.info(" -- Cannot drop the efficiency of component \""
					+ component.getUid() + "\" from tenant \""+ component.getTenant().getName());
			System.out.print("\n -- Cannot drop the efficiency of component \""
					+ component.getUid() + "\" from tenant \""+ component.getTenant().getName());
			return false;
		}
	}

}
