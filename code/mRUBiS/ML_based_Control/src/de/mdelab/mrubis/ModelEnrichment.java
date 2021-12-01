package de.mdelab.mrubis;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Random;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Server;
import de.mdelab.morisia.comparch.Tenant;

public class ModelEnrichment {

	private final static URI BASE = URI.createFileURI(new File(".")
			.getAbsolutePath());
	private static ResourceSet RS = new ResourceSetImpl();

	public static void main(String[] args) {

		/*
		 * Load model
		 */
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put(
				"comparch", new XMIResourceFactoryImpl());
		ComparchPackage.eINSTANCE.eClass();

		URI architectureResourceURI = URI.createFileURI(
				"model/mRUBiS-1shop.comparch").resolve(BASE);

		Resource architectureResource = loadInstance(architectureResourceURI);
		Architecture eArchitecture = (Architecture) architectureResource
				.getContents().get(0);

		Annotations annotations = eArchitecture.getAnnotations();
		if (annotations == null) {
			annotations = ComparchFactory.eINSTANCE.createAnnotations();
			eArchitecture.setAnnotations(annotations);
		}

		/*
		 * Enrich model
		 */
		enrichWithReliability(eArchitecture, true);
		//enrichWithDifferentCriticality(eArchitecture);
		// enrichWithSameCriticality(eArchitecture);
	 // enrichWithExteremeCriticality(eArchitecture);
		enrichWithRandomCriticality(eArchitecture);
		enrichWithImportance(eArchitecture);
		enrichWithADT(eArchitecture);
	 checkZeroADT(eArchitecture);
		enrichShopReplicaFactor(eArchitecture);
		enrichReplicaType(eArchitecture);
		enrichReplicaComponent(eArchitecture); // should be set after type
       enrichServervalues(eArchitecture);
		
		
		
		/*
		 * Save model
		 */
		URI resultURI = URI.createFileURI("model/enriched/mRUBiS-1shop_enriched.comparch").resolve(BASE);
		//URI resultURI = URI.createFileURI("model/mRUBiS-100ML_test_with_ADT.comparch").resolve(BASE);
		
				
		architectureResource.setURI(resultURI);

		try {
			architectureResource.save(Collections.EMPTY_MAP);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private static void enrichShopReplicaFactor(Architecture eArchitecture) {
		double Minval=10;
		double Maxval=30;
		 
		
		for(Tenant shop:eArchitecture.getTenants())
		{Random r = new Random();
		double replicaFactor = Minval + (Maxval - Minval) * r.nextDouble();
			shop.setReplicaFactor(replicaFactor);
			
		}
		
	}

	private static void         enrichServervalues(Architecture eArchitecture) {
		int counter = 0;
		int serverNumber=eArchitecture.getCloud().getServers().size();

		for (Server server : eArchitecture.getCloud().getServers()) {

			// set consumption & cost values to zero
			server.setCPUConsumption(0);
			server.setMemoryConsumption(0);
			server.setStorageConsumption(0);
			server.setCost(0);
			
			
			//Server CPU consumption factor for each component type 
			double MinFactor=1;
			double MaxFactor=10;
			 Random r = new Random();
			double cpuFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			server.setCpuFactor(cpuFactor);
			
			//Server memory consumption factor for each component type 
			 MinFactor=1;
			 MaxFactor=15;
			 r = new Random();
			double memoryFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			server.setMemoryFactor(memoryFactor);
			
			//Server storage consumption factor for each component type 
			 MinFactor=20;
			 MaxFactor=40;
			 r = new Random();
			double storageFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			server.setStorageFactor(storageFactor);
            
			//randomly set Capacity values
			if (counter<(serverNumber/3)) //slow server = low storage
			{
				server.setCPUCapacity(4E9);
				server.setMemoryCapacity(4E9);
				server.setStorageCapacity(1E9);
			}

			else if (counter<(2*serverNumber/3)) //Medium server
			{
				server.setCPUCapacity(16E9);
				server.setMemoryCapacity(64E9);
				server.setStorageCapacity(1E10);
			}

			else //fast server +  large storage
			{
				server.setCPUCapacity(96E9);
				server.setMemoryCapacity(384E9);
				server.setStorageCapacity(1E11);

			}
counter ++;
		}
	}

	private static void enrichReplicaComponent(Architecture eArchitecture) {
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				
				 //Request per component
				int maxReq=500;
				int minReq=50;
				Random rn = new Random();
				double request = rn.nextInt((maxReq - minReq) + 1) + minReq;
				component.setRequest(request);
				component.setOriginalRequest(request);
				
				//Inuse replica -- requires Request and Saturation to be set first
				double inUseReplica=Math.ceil(component.getType().getSatPoint()*component.getRequest());
				component.setInUseReplica(inUseReplica);
				
				//Performance = Pmax
				double performance=component.getType().getPerformanceMax();
				double RealP=performance*Math.tanh((4/component.getType().getSatPoint())*inUseReplica/request);
				component.setPerformance(performance);
				double Deltap=performance-RealP;
				//System.out.print("\n Delta P: "+ Deltap);
				//Efficiency=pmax/iuse*adt*cost
				double efficiecincy=component.getPerformance()
						/(component.getInUseReplica()
						*component.getType().getReplicaADT()
						*component.getType().getReplicaCost());
		        component.setEfficiency(efficiecincy);
		        
	
		        
		 
			}
		}
		
		
	}

	private static void enrichReplicaType(Architecture eArchitecture) {
		for (ComponentType componentType : eArchitecture
				.getComponentTypes()) {
			
			//Saturation point		
			double MinSP=0.02;
			double MaxSP=0.06;
			Random r = new Random();
			double saturationPoint = MinSP + (MaxSP - MinSP) * r.nextDouble();
			componentType.setSatPoint(saturationPoint);
			
			//Max performance of each component type
			double MinPMAX=5;
			double MaxPMAX=20;
			 r = new Random();
			 double Pmax = MinPMAX + (MaxPMAX - MinPMAX) * r.nextDouble();
			componentType.setPerformanceMax(Pmax);
			
			//cost of each replica
			double MinCOST=1;
			double MaxCOST=1.5;
			 r = new Random();
			double repilicaCost = MinCOST + (MaxCOST - MinCOST) * r.nextDouble();
			componentType.setReplicaCost(repilicaCost);
			
			//average deployment time of each replica
			double MinADT=1;
			double MaxADT=1.5;
			 r = new Random();
			double replicaADT = MinADT + (MaxADT - MinADT) * r.nextDouble();
			componentType.setReplicaADT(replicaADT);
			
			//replica CPU consumption factor for each component type 
			double MinFactor=1;
			double MaxFactor=3;
			 r = new Random();
			double cpuFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			componentType.setCpuFactor(cpuFactor);
			
			//replica memory consumption factor for each component type 
			 MinFactor=1;
			 MaxFactor=3;
			 r = new Random();
			double memoryFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			componentType.setMemoryFactor(memoryFactor);
			
			//replica storage consumption factor for each component type 
			 MinFactor=1;
			 MaxFactor=3;
			 r = new Random();
			double storageFactor = MinFactor + (MaxFactor - MinFactor) * r.nextDouble();
			componentType.setStorageFactor(storageFactor);
			
			
		
		}
			
	}

	private static void enrichWithImportance(Architecture eArchitecture) {
		
		//Normally set importance of Tenants
		for (Tenant shop : eArchitecture.getTenants()) {
			
				  
					int maxC=25;
					int minC=10;
					Random rn = new Random();
					 int imptnc = rn.nextInt((maxC - minC) + 1) + minC;
					 shop.setImportance(imptnc);
					
				
				//component.setImportance(0);
			
		}
		
	}

	private static void enrichWithReliability(Architecture eArchitecture,
			boolean makeGoogleBest) {
		if (makeGoogleBest) {
			for (ComponentType componentType : eArchitecture
					.getComponentTypes()) {
				if (componentType.getName().startsWith("Google")) {
					componentType.setReliability(0.90);
				} else if (componentType.getName().startsWith("Facebook")) {
					componentType.setReliability(0.85);
				} else if (componentType.getName().startsWith("Twitter")) {
					componentType.setReliability(0.85);
				} else if (componentType.getName().equals(
						"Authentication Service")) {
					componentType.setReliability(0.85);
				} else {
					componentType.setReliability(0.5);
				}
			}
		} else {
			for (ComponentType componentType : eArchitecture
					.getComponentTypes()) {
				if (componentType.getName().startsWith("Google")) {
					Random rn=new Random();
					double gRN=rn.nextDouble();
					componentType.setReliability(gRN);
				} else if (componentType.getName().startsWith("Facebook")) {
					Random rn=new Random();
					double fRN=rn.nextDouble();
					componentType.setReliability(fRN);
				} else if (componentType.getName().startsWith("Twitter")) {
					Random rn=new Random();
					double tRN=rn.nextDouble();
					componentType.setReliability(tRN);
				} else if (componentType.getName().equals(
						"Authentication Service")) {
					Random rn=new Random();
					double asRN=rn.nextDouble();
					componentType.setReliability(asRN);
				} else {
					componentType.setReliability(1);
				}
			}
		}

	}

	private static void enrichWithDifferentCriticality(
			Architecture eArchitecture) {

		// set criticality and importance of each component to 1
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				if (component.getType().getName().contains("Authentication")) {
					component.setCriticality(7.0);
				} else if (component.getType().getName()
						.equals("Reputation Service")) {
					component.setCriticality(1.0);
				} else if (component.getType().getName()
						.contains("Item Filter")) {
					component.setCriticality(2.0);
				} else {
					component.setCriticality(3.0);
				}
				component.setImportance(0);
			}
		}

	}

	private static void enrichWithExteremeCriticality(
			Architecture eArchitecture) {

		// set criticality and importance of each component to 1
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				if (component.getType().getName().contains("Authentication")) {
					component.setCriticality(17.0);
					component.setImportance(5.0);
				} else if (component.getType().getName()
						.equals("Reputation Service")) {
					component.setCriticality(1.0);
					component.setImportance(3);
				} else if (component.getType().getName()
						.contains("Item Filter")) {
					component.setCriticality(2.0);
					component.setImportance(0);
				} else {
					component.setCriticality(3.0);
					component.setImportance(1);
				}
				//component.setImportance(0);
			}
		}

	}
	private static void enrichWithRandomCriticality(
			Architecture eArchitecture) {
		

		// RANDOMLY set criticality  each component type
		
			for (ComponentType componentType : eArchitecture
					.getComponentTypes()) {
				/*if (component.getType().getName().contains("Authentication")) {
					int maxC=25;
					int minC=16;
					Random rn = new Random();
					 int crit = rn.nextInt((maxC - minC) + 1) + minC;
					component.setCriticality(crit);
					component.setImportance(5.0);
				}*/ 
					int maxC=30;
					int minC=10;
					Random rn = new Random();
					 int crit = rn.nextInt((maxC - minC) + 1) + minC;
					componentType.setCriticality(crit);
					
				System.out.print("\n Criticality of Type "+ componentType.eClass().getName()+" is"+componentType.getCriticality());
				//component.setImportance(0);
			}
		}

	
	private static void enrichWithADT(
			Architecture eArchitecture) {

		// set ADT of each component to Normally Distributed values between [1 , 1.5]
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
			 
				    double rangeMax=1.5;
					double  rangeMin=1;
					Random rn = new Random();
					 double adt =rangeMin + (rangeMax - rangeMin) * rn.nextDouble();
					component.setADT(adt);
					//System.out.print("adt generated : "+ component.getADT() +"\n");
				
				//component.setImportance(0);
			}
		}

	}
	private static void checkZeroADT(
			Architecture eArchitecture) {

		
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				if(component.getADT()==0){
				System.out.print("\n  ADT Zero found! " +"\n");
				    double rangeMax=1.5;
					double  rangeMin=1;
					Random rn = new Random();
					 double adt =rangeMin + (rangeMax - rangeMin) * rn.nextDouble();
					component.setADT(adt);
					
				}
				//component.setImportance(0);
			}
		}

	}
	private static void enrichWithSameCriticality(Architecture eArchitecture) {
		// set criticality and importance of each component to 1
		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				component.setCriticality(1);
				component.setImportance(0);
			}
		}
	}

	private static Resource loadInstance(URI uri) {
		// ResourceSet rs = new ResourceSetImpl();
		Resource resource = RS.getResource(uri, true);
		return resource;
	}
}
