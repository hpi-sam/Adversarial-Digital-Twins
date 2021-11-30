package LoadReplicaRequirement;

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
import de.mdelab.morisia.comparch.Tenant;

/*
 * 
 * Generates Load for Shops and computes the proper number of Replicas for each Component within Shops to achieve the required Performance 
 * Performance is computed with respect to the SatiratingPoint of the ComponentType, set in the ModelEnrichment.java
 * 
 * 
 */
public class ShopLoadGenerator {

	private final static URI BASE = URI.createFileURI(new File(".").getAbsolutePath());
	private static ResourceSet RS = new ResourceSetImpl();

	public static void randomLoadGenerator() {

		/*
		 * Load model
		 */
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("comparch", new XMIResourceFactoryImpl());
		ComparchPackage.eINSTANCE.eClass();

		//URI architectureResourceURI = URI.createFileURI("./model/mRUBiS-50shop20Ser_enriched.comparch").resolve(BASE);
		URI architectureResourceURI = URI.createFileURI("./model/mRUBiS-100shop20Ser_enriched.comparch").resolve(BASE);

		Resource architectureResource = loadInstance(architectureResourceURI);
		Architecture eArchitecture = (Architecture) architectureResource.getContents().get(0);

		Annotations annotations = eArchitecture.getAnnotations();
		if (annotations == null) {
			annotations = ComparchFactory.eINSTANCE.createAnnotations();
			eArchitecture.setAnnotations(annotations);
		}

		/*
		 * 
		 * 
		 */
		double Minval = 10;
		double Maxval = 50;

		Random r = new Random();
		double request = 0;

		for (Tenant shop : eArchitecture.getTenants()) {
			r = new Random();
			request = Math.floor(Minval + (Maxval - Minval) * r.nextDouble());
			shop.setRequest(request);
			// System.out.print("\n Request " + request);

			setReplicaNumbers(shop);
			setReplicaRequirement(shop);

		}
		/*
		 * Save model
		 */
		//URI resultURI = URI.createFileURI("model/enriched/mRUBiS-50shop20Ser_enrichedwithLoad.comparch").resolve(BASE);
		// URI resultURI =
		// URI.createFileURI("model/mRUBiS-100ML_test_with_ADT.comparch").resolve(BASE);
		URI resultURI = URI.createFileURI("model/enriched/mRUBiS-100shop20Ser_enrichedwithLoad.comparch").resolve(BASE);
		
		architectureResource.setURI(resultURI);

		try {
			architectureResource.save(Collections.EMPTY_MAP);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private static void setReplicaRequirement(Tenant shop) {
		for (Component component : shop.getComponents()) {
			// cpu requirement : divided equally between all replicas+ 1
			// component
			double CpuRequirement = shop.getReplicaFactor() * component.getType().getCpuFactor() * shop.getRequest()
					/ (1 + component.getInUseReplica()); // ai=shop.getReplicaFactor()*
															// component.getType().getCpuFactor()
			component.getReplica().setCPURequirement(CpuRequirement);

			// memory requirement : divided equally between all replicas+ 1
			// component
			double MemoryRequirement = shop.getReplicaFactor() * component.getType().getMemoryFactor()
					* shop.getRequest() / (1 + component.getInUseReplica()); // bi=shop.getReplicaFactor()*
																				// component.getType().getMemoryFactor()
			component.getReplica().setMemoryRequirement(MemoryRequirement);

			// storage requirement : divided equally between all replicas+ 1
			// component
			double storageRequirement = shop.getReplicaFactor() * component.getType().getStorageFactor()
					* shop.getRequest() / (1 + component.getInUseReplica()); // ti=shop.getReplicaFactor()*
																				// component.getType().getStorageFactor()
			component.getReplica().setStorageRequirement(storageRequirement);

		}

	}

	private static void setReplicaNumbers(Tenant shop) {
		for (Component component : shop.getComponents()) {
			// Inuse replica -- requires Request and Saturation to be set first
			double inUseReplica = Math.ceil(component.getType().getSatPoint() * shop.getRequest());
			component.setInUseReplica(inUseReplica);
		}

	}

	private static Resource loadInstance(URI uri) {
		// ResourceSet rs = new ResourceSetImpl();
		Resource resource = RS.getResource(uri, true);
		return resource;
	}

	/*
	 * Load generation based on shop load profiles
	 */
	public static void profileLoadGenerator(Architecture architecture) {

		// TODO

	}

}
