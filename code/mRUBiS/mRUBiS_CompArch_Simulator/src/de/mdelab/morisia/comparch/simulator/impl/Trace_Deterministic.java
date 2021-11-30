package de.mdelab.morisia.comparch.simulator.impl;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

public class Trace_Deterministic implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;


	public  Trace_Deterministic(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;

		this.eArchitecture = eArchitecture;

	}

	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
		
		
		

		//if (runCount%3==0) 
		if (runCount%10==0) 
			//6 CF
			{ System.out.print("\n 10");
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(0, 1)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(9, 1)));
				
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(2, 11)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(5, 13)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(7, 15)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.getComponent(8, 7).getProvidedInterfaces().get(0)));
				
				
				
			} else if (runCount%10 ==1) 
			// 4 CF
			{System.out.print("\n 1");
				// Google is best as defined in the model
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(0, 4)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(3, 3)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(6, 2)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(9, 11)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(7, 0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(5, 15)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(4, 14)));
				
			} else if (runCount%10==2) 
				// 7 CF
			{System.out.print("\n 2");
				Component component = this.getComponent(6, 3);
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						component.getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(1, 3)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.findMostCriticalComponent(7).getProvidedInterfaces()
						.get(0)));
				//injections.add(new Injection<Component>(IssueType.CF3, this
						//.findLeastCriticalComponent(68)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(9, 5)));
				//injections.add(new Injection<Component>(IssueType.CF3, this
						//.findLeastCriticalComponent(72)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(4, 3)));
				
			} else if (runCount%10 == 3)
			//5 CF
				{System.out.print("\n 3");
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.findMostCriticalComponent(8).getProvidedInterfaces()
						.get(0)));
				Component component = this.getComponent(1, 3);
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						component.getProvidedInterfaces().get(0)));
				 component = this.getComponent(3, 7);
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						component.getProvidedInterfaces().get(0)));
				 component = this.getComponent(7, 7);
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						component.getProvidedInterfaces().get(0)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.findMostCriticalComponent(2).getProvidedInterfaces()
						.get(0)));
			
			} else if (runCount%10 == 4)
			// 10 CF
				{System.out.print("\n 4");
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(5, 3)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(0, 1)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(8, 3)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(1, 2)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(4, 4)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(9, 5)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(3, 6)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(6, 1)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(2, 7)));
			
				} else if (runCount%10 == 5) {
					System.out.print("\n 5");
				
						injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(8, 9)));
					Component component = this.getComponent(1, 6);
					injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
							component.getProvidedInterfaces().get(0)));
					
			
			}
			
		else if (runCount%10 == 6) {
			System.out.print("\n 6");
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(4, 4)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.getComponent(5, 4).getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(6, 4)));
			
		} else if (runCount%10 == 7) {
			System.out.print("\n 7");
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(2, 10)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(6, 14)));
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(5, 3)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getComponent(9, 7)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.getComponent(8, 7).getProvidedInterfaces().get(0)));
			
			} else if (runCount%10 == 8) {
				System.out.print("\n 8");
				injections.add(new Injection<Component>(IssueType.CF1, this
						.getComponent(6, 6)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.getComponent(5, 6).getProvidedInterfaces().get(0)));
				
			} else if (runCount%10 == 9) {
				System.out.print("\n 9");
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.findMostCriticalComponent(7).getProvidedInterfaces()
						.get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.findLeastCriticalComponent(8)));
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
						.findMostCriticalComponent(5).getProvidedInterfaces()
						.get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.findLeastCriticalComponent(2)));
			
			}  
	
		return injections;
	}

	@Override
	public void notifyAboutSuccess(
			List<Injection<? extends ArchitecturalElement>> injections) {
		for (Injection<? extends ArchitecturalElement> i : injections) {
			if (!i.isSuccess()) {
				String message = "The simulator could not successfully inject a "
						+ i.getIssueType()
						+ " issue to element "
						+ i.getTarget();
				System.err.println(message);
				throw new RuntimeException(message);
			}
		}
	}

	private Component getComponent(int tenantNumber, int componentNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		return tenant.getComponents().get(componentNumber);
	}
	private int selectRandomComponent() {
		Tenant tenant = this.eArchitecture.getTenants().get(0);
		if (tenant != null) {
			
			Component component = null;
			int componentNumber =-1;
			while (component == null) {
				 componentNumber = SimulatorUtil.randInt(0, tenant
						.getComponents().size() - 1);
				// System.out.println("\n comp number is   " +componentNumber);
				component = tenant.getComponents().get(componentNumber);
				
				
				if (component.getIssues().size()>0)
				{
					component = null;}
				else if (component.getType().getName()
						.equals("Future Sales Item Filter")) {
					// last filter of the pipe should bot be affected by an
					// issue.
					//component = null;
				}
			}
			return componentNumber;
		} else {
			return -1;
		}
	}
	


	private Component findLeastCriticalComponent(int tenantNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		double selectedCriticality = Double.MAX_VALUE;
		Component selectedComponent = null;
		for (Component component : tenant.getComponents()) {
			double criticality = component.getType().getCriticality();
			if (criticality < selectedCriticality) {
				selectedCriticality = criticality;
				selectedComponent = component;
			}
		}
		return selectedComponent;
	}
	

	private Component findMostCriticalComponent(int tenantNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		double selectedCriticality = Double.MIN_VALUE;
		Component selectedComponent = null;
		for (Component component : tenant.getComponents()) {
			double criticality = component.getType().getCriticality();
			if (criticality > selectedCriticality) {
				selectedCriticality = criticality;
				selectedComponent = component;
			}

		}
		return selectedComponent;
	}
	
	
}
