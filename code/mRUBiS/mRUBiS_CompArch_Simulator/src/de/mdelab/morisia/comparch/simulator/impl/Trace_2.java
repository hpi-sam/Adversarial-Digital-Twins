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

public class Trace_2 implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;


	public  Trace_2(IssueType[] issueTypes,
			Architecture eArchitecture) {
		this.issueTypes = issueTypes;

		this.eArchitecture = eArchitecture;

	}

	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(
			int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
		
		
		

			
		
	
		  //10 shops
		  for(int shop=0; shop<10; shop++)
		  
		  { 	int max=1;// 1 CF per run
			int min=0;
			Random rnd = new Random();
			  int injectionNumber = rnd.nextInt((max - min) + 1) + min;
			  
			  int typeMax=4;
			int typeMin=1;
			Random typrnd = new Random();
	 
			for (int i=0; i<injectionNumber; i++) // 0 or 1 CF per shop
				
				{int component = this.selectRandomComponent();
				 int issuenumber = rnd.nextInt((typeMax - typeMin) + 1) + typeMin;
				 
				 if(issuenumber==0||component==-1)
				 {//noCF
					 System.out.println("\n No CF Injected \n" );
					 }
				 else if(issuenumber==1)
				 {//CF1
					 injections.add(new Injection<Component>(IssueType.CF1, this
								.getComponent(shop, component)));
					 System.out.println("\n Injected CF1\n" );
					 }
				 else if(issuenumber==2)
				 {//CF2
					 injections.add(new Injection<ProvidedInterface>(IssueType.CF2, this
								.getComponent(shop, component).
							 getProvidedInterfaces().get(0)));
					 System.out.println("\n Injected CF2\n" );
					 }
				 else if(issuenumber==3)
				 {//CF3
					 injections.add(new Injection<Component>(IssueType.CF3, this
								.getComponent(shop, component)));
					 System.out.println("\n Injected CF3\n" );
					 }
				 else if(issuenumber==4)
				 {//CF5
					 injections.add(new Injection<Component>(IssueType.CF5, this
								.getComponent(shop, component)));
					 System.out.println("\n Injected CF5\n" );
					 }
				 
				}
		
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
	



	

}
