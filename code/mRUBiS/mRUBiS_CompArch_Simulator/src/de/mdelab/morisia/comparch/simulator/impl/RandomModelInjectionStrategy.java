package de.mdelab.morisia.comparch.simulator.impl;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

public class RandomModelInjectionStrategy implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;


	public RandomModelInjectionStrategy(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;
		System.out
		.println("Static injection strategy for experiments may take the following issues into account: "
				+ Arrays.toString(this.issueTypes));
		this.eArchitecture = eArchitecture;
		

	}

	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount ) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
		
		
			
			/*     CF_Count=1  && TenntCount =100 ---
			 * 
			 * ********************************************************************************
			 */
		// 1 CF
		/*
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
		
			injections.add(new Injection<Component>(IssueType.CF3, this.findMostCriticalComponent(50)));
			
		*/
		
		
		/*     CF_Count=10  && TenntCount =100 ---
		 * 
		 * ********************************************************************************
		 */
	//10
	
	/*
				this.updateReliabiliy_G_1();
				this.updateReliabiliy_FB_1();
			
					 
				injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(3)));
				injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(10)));
				injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(17)));
				
			
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(22, 1)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(29, 6)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(34, 13)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(43, 15)));
			
		
				this.updateReliabiliy_G_1();
				this.updateReliabiliy_TW_1();
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(58).getProvidedInterfaces()
					.get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(62).getProvidedInterfaces()
					.get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(77).getProvidedInterfaces()
					.get(0)));
			
			*/
			
			/*     CF_Count=100  && TenntCount =100 ---
			 * 
			 * ********************************************************************************
			 */
		
		//100
		/*
			for (int i=0; i<10; i++) 
			{
				this.updateReliabiliy_G_1();
				this.updateReliabiliy_FB_1();
				this.updateReliabiliy_TW_1();
				
	    // 30 times CF3
			injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(i)));
			injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(i+65)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(i+45,5)));
		// 40 times CF1
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i+73, 11)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i+20, 12)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i+34, 13)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i+6, 14)));
		// 30 times CF2
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i+87, 10).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i+55, 9).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i+22, 1).getProvidedInterfaces().get(0)));
		
			}
			*/
			/*     CF_Count=1000  && TenntCount =100 ---
			 * 
			 * ********************************************************************************
			 */
		
		//1000
		this.updateReliabiliy_G_1();
		this.updateReliabiliy_FB_1();
		this.updateReliabiliy_TW_1();
			for (int i=0; i<100; i++) 
			{
	    // 30 times CF3
			injections.add(new Injection<Component>(IssueType.CF3, this.getAuthenticationComponent(i)));
					
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(i,4)));
		injections.add(new Injection<Component>(IssueType.CF3, this
				.getComponent(i,5)));
		// 40 times CF1
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i, 11)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i, 12)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i, 13)));
		injections.add(new Injection<Component>(IssueType.CF1, this
				.getComponent(i, 14)));
		// 30 times CF2
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i, 10).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i, 9).getProvidedInterfaces().get(0)));
		injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
				.getComponent(i, 1).getProvidedInterfaces().get(0)));}
		
			
			
			
			/*injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(count).getProvidedInterfaces()
					.get(0)));*/
			
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

	private Component getAuthenticationComponent(int tenantNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		Component authenticationService = null;
		for (Component c : tenant.getComponents()) {
			if (c.getType().getName().equals("Authentication Service")) {
				authenticationService = c;
				break;
			}
		}
		return authenticationService;
	}

	private void updateReliabiliy_FB_1() {
		for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
			String componentTypeName = ct.getName();
			if (componentTypeName.startsWith("Facebook")) {
				ct.setReliability(0.99);
			}
		}
	}

	private void updateReliabiliy_G_1() {
		for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
			String componentTypeName = ct.getName();
			if (componentTypeName.startsWith("Google")) {
				ct.setReliability(0.04);
			}
		}
	}

	private void updateReliabiliy_TW_1() {
		for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
			String componentTypeName = ct.getName();
			if (componentTypeName.startsWith("Twitter")) {
				ct.setReliability(0.98);
			}
		}
	}

	private Component findMostCriticalComponent(int tenantNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		double selectedCriticality = Double.MIN_VALUE;
		Component selectedComponent = null;
		for (Component component : tenant.getComponents()) {
			double criticality = component.getCriticality();
			if (criticality > selectedCriticality) {
				selectedCriticality = criticality;
				selectedComponent = component;
			}

		}
		return selectedComponent;
	}

	private Component findLeastCriticalComponent(int tenantNumber) {
		Tenant tenant = this.eArchitecture.getTenants().get(tenantNumber);
		double selectedCriticality = Double.MAX_VALUE;
		Component selectedComponent = null;
		for (Component component : tenant.getComponents()) {
			double criticality = component.getCriticality();
			if (criticality < selectedCriticality) {
				selectedCriticality = criticality;
				selectedComponent = component;
			}
		}
		return selectedComponent;
	}

}
