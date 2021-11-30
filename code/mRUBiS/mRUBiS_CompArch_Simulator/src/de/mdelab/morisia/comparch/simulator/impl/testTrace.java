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

public class testTrace implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;


	public  testTrace(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;

		this.eArchitecture = eArchitecture;

	}

	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
		
		
		injections.add(new Injection<Component>(IssueType.CF1, this.findMostCriticalComponent(0)));
		// injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this.getComponent(0, 2).
			//	 getProvidedInterfaces().get(0)));
/*if(runCount%10==0) {
			injections.add(new Injection<Component>(IssueType.CF3, this.getComponent(0, 2)));
			
		}
		
		else{
			int modelSize=this.eArchitecture.getTenants().size();
		
		int max=1;// STATIC: 2 OURS:3 SOLVER:7 
		int min=1;
		Random rnd = new Random();
		  int injectionNumber = rnd.nextInt((max - min) + 1) + min;
		 
		  
		  int typeMax=4;
			int typeMin=1;
			Random typrnd = new Random();
	 
			for (int i=0; i<injectionNumber; i++) // 0<=number<18
				//for (int i=0; i<1; i++) // 0<=number<18
				{Component component = this.selectRandomComponent();
				 int issuenumber = rnd.nextInt((typeMax - typeMin) + 1) + typeMin;
				 
				 if(issuenumber==0||component==null)
				 {//noCF
					 System.out.println("\n No CF Injected \n" );
					 }
				 else if(issuenumber==1)
				 {//CF1
					 injections.add(new Injection<Component>(IssueType.CF1, component));
					 System.out.println("\n Injected CF1\n" );
					 }
				 else if(issuenumber==2)
				 {//CF2
					 injections.add(new Injection<ProvidedInterface>(IssueType.CF2, component.
							 getProvidedInterfaces().get(0)));
					 System.out.println("\n Injected CF2\n" );
					 }
				 else if(issuenumber==3)
				 {//CF3
					 injections.add(new Injection<Component>(IssueType.CF3, component));
					 System.out.println("\n Injected CF3\n" );
					 }
				 else if(issuenumber==4)
				 {//CF5
					 injections.add(new Injection<Component>(IssueType.CF5, component));
					 System.out.println("\n Injected CF5\n" );
					 }
				 
				}
		
		}
	
			*/
		/*     CF_Count=1000  && TenntCount >=1000
		 * 
		 * ********************************************************************************
		 */
 /*
			        injections.add(new Injection<Component>(IssueType.CF5, 
					        this.getComponent(17, 10)));
					injections.add(new Injection<Component>(IssueType.CF3, 
							this.getComponent(8,2)));
					injections.add(new Injection<Component>(IssueType.CF1, 
							this.getComponent(12, 10)));
					injections.add(new Injection<ProvidedInterface>(IssueType.CF2, 
							this.getComponent(1, 5).getProvidedInterfaces().get(0)));


				*/
				
				/*     CF_Count=1000  && TenntCount =100 --FOR STATIC ONLY
				 * 
				 * ********************************************************************************
				 */
				/*
				
					if (runCount == 1) {
							for (int i=0; i<100; i++) // 1000 times CF1

							{
								injections.add(new Injection<Component>(IssueType.CF1, this
									.getComponent(i, 1)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 2)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 4)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 5)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 6)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 10)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 11)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 12)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 13)));
								injections.add(new Injection<Component>(IssueType.CF1, this
										.getComponent(i, 14)));

							}

							
						} else if (runCount == 2) {
							for (int i=0; i<100; i++) //  1000 times CF3
							{

								
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,6)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,5)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,11)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,10)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,8)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,12)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,1)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,13)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,4)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,14)));
								
								
								
							}
							
							
							
						} else if (runCount == 3) {
							for (int i=0; i<100; i++) // 100 times CF2

								injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
										.findMostCriticalComponent(i).getProvidedInterfaces()
										.get(0)));

							for (int i=0; i<100; i++) // 100 times CF3

								injections.add(new Injection<Component>(IssueType.CF3, this
										.findLeastCriticalComponent(i)));

							for (int i=0; i<100; i++) // 70 times CF1
							{
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,13)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,4)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,16)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,17)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,2)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,11)));
								injections.add(new Injection<Component>(IssueType.CF3, this
										.getComponent(i,5)));

								
								
								}*/
								
		/*     Mixed scenarios   // runCount starts with 1
		 * 
		 * ********************************************************************************
		 */
		
		/*				
		
		if (runCount == 3) 
		//6 CF
		{ System.out.print("1st mape");
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(0, 1)));
			this.updateReliabiliy_FB_2();
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(85)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(90)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(75, 3)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(77, 7)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(80, 7).getProvidedInterfaces().get(0)));
		} else if (runCount == 4)
		// 4 CF
		{System.out.print("2nd mape");
			// Google is best as defined in the model
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(2)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(10)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(20)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(30)));
			
		} else if (runCount == 5) 
			// 7 CF
		{System.out.print("3rd mape");
			Component component = this.getComponent(10, 3);
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					component.getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(15, 3)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(67).getProvidedInterfaces()
					.get(0)));
			//injections.add(new Injection<Component>(IssueType.CF3, this
					//.findLeastCriticalComponent(68)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(70).getProvidedInterfaces()
					.get(0)));
			//injections.add(new Injection<Component>(IssueType.CF3, this
					//.findLeastCriticalComponent(72)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(11, 3)));
			
		} else if (runCount == 6)
		//5 CF
			{System.out.print("4th mape");
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(18).getProvidedInterfaces()
					.get(0)));
			Component component = this.getComponent(10, 3);
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					component.getProvidedInterfaces().get(0)));
			 component = this.getComponent(30, 7);
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					component.getProvidedInterfaces().get(0)));
			 component = this.getComponent(77, 7);
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
					component.getProvidedInterfaces().get(0)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(22).getProvidedInterfaces()
					.get(0)));
		
		} else if (runCount == 2)
		// 10 CF
			{System.out.print("5th mape");
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(25, 3)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(0, 1)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(28, 3)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(30, 2)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(40, 4)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(49, 5)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(53, 6)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(59, 1)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(68, 9)));
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(70, 7)));
		
			} else if (runCount == 1) {
			
				System.out.print("mixed mape");
			
					injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(68, 9)));
					this.updateReliabiliy_FB_1();
					
					
				Component component = this.getComponent(10, 6);
				injections.add(new Injection<ProvidedInterface>(IssueType.CF2,
						component.getProvidedInterfaces().get(0)));
				injections.add(new Injection<Component>(IssueType.CF3, this
						.getAuthenticationComponent(30)));
		
		}
		
	else if (runCount == 7) {
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(40, 4)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(45, 4).getProvidedInterfaces().get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(50, 4)));
		} else if (runCount == 8) {
			this.updateReliabiliy_G_1();
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(52)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(54)));
		} else if (runCount == 9) {
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(60, 6)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(65, 6).getProvidedInterfaces().get(0)));
		} else if (runCount == 10) {
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(67).getProvidedInterfaces()
					.get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.findLeastCriticalComponent(68)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.findMostCriticalComponent(70).getProvidedInterfaces()
					.get(0)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.findLeastCriticalComponent(72)));
		} else if (runCount == 11) {
			injections.add(new Injection<Component>(IssueType.CF1, this
					.getComponent(75, 3)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getComponent(77, 7)));
			injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
					.getComponent(80, 7).getProvidedInterfaces().get(0)));
		} else if (runCount == 12) {
			this.updateReliabiliy_FB_2();
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(85)));
			injections.add(new Injection<Component>(IssueType.CF3, this
					.getAuthenticationComponent(90)));
		}*/
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
	private Component selectRandomComponent() {
		Tenant tenant = this.eArchitecture.getTenants().get(0);
		if (tenant != null) {
			
			Component component = null;
			while (component == null) {
				int componentNumber = SimulatorUtil.randInt(0, tenant
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
					component = null;
				}
			}
			return component;
		} else {
			return null;
		}
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
				ct.setReliability(0.92);
			}
		}
	}

	private void updateReliabiliy_G_1() {
		for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
			String componentTypeName = ct.getName();
			if (componentTypeName.startsWith("Google")) {
				ct.setReliability(0.94);
			}
		}
	}

	private void updateReliabiliy_FB_2() {
		for (ComponentType ct : this.eArchitecture.getComponentTypes()) {
			String componentTypeName = ct.getName();
			if (componentTypeName.startsWith("Facebook")) {
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
