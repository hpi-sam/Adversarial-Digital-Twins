package de.mdelab.morisia.selfhealing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import de.mdelab.mlcallactions.MlcallactionsPackage;
import de.mdelab.mlexpressions.MLExpression;
import de.mdelab.mlexpressions.MlexpressionsPackage;
import de.mdelab.mlsdm.Activity;
import de.mdelab.mlsdm.MlsdmPackage;
import de.mdelab.mlsdm.interpreter.MLSDMExpressionInterpreterManager;
import de.mdelab.mlsdm.interpreter.MLSDMInterpreter;
import de.mdelab.mlsdm.interpreter.MLSDMCallActionInterpreter;
import de.mdelab.mlsdm.interpreter.notifications.MLSDMOutputStreamNotificationReceiver;
import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.sdm.interpreter.core.SDMException;
import de.mdelab.expressions.interpreter.core.variables.Variable;
import de.mdelab.expressions.interpreter.ocl.OCLExpressionInterpreter;

public class BatchTest {

	public static void main(String[] args) throws SDMException {

		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put(
				"comparch", new XMIResourceFactoryImpl());
		ComparchPackage.eINSTANCE.eClass();
		URI base = URI.createFileURI(new File(".").getAbsolutePath());

		URI architectureResourceURI = URI
				.createFileURI("model/mRUBiS.comparch").resolve(base);

		Resource architectureResource = loadInstance(architectureResourceURI);
		Architecture eArchitecture = (Architecture) architectureResource
				.getContents().get(0);

		Annotations annotations = ComparchFactory.eINSTANCE.createAnnotations();
		eArchitecture.setAnnotations(annotations);
		/*
		 * Working with the interpreter requires at first the registration of
		 * the story diagram resources.
		 */
		initializeStoryDiagrams();

		Activity activity = getStoryDiagramActivityFromFile("model/batch/A-CF2-batch.mlsdm");

		MLSDMInterpreter interpreter = getStoryDiagramInterpreter();

		/*
		 * Register as many languages as you want
		 */
		interpreter.getExpressionInterpreterManager()
				.registerExpressionInterpreter(
						new OCLExpressionInterpreter<MLExpression>(), "OCL",
						"1.0");
		interpreter.getExpressionInterpreterManager()
				.registerExpressionInterpreter(new MLSDMCallActionInterpreter(),
						"CallActions", "1.0");

		/*
		 * Parameter stuff
		 */
		Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
		Variable<EClassifier> param1 = new Variable<EClassifier>(
				"architecture", architectureResource.getContents().get(0)
						.eClass(), architectureResource.getContents().get(0));
		parameters.add(param1);
		Variable<EClassifier> param2 = new Variable<EClassifier>("annotations",
				annotations.eClass(), annotations);
		parameters.add(param2);

		/*
		 * Execute diagram and get results
		 */
		Map<String, Variable<EClassifier>> retValues;
		retValues = interpreter.executeActivity(activity, parameters);

		/*
		 * Save stuff
		 */
		URI resultURI = URI.createFileURI("model/mRUBiS_RESULT.comparch")
				.resolve(base);
		architectureResource.setURI(resultURI);
		try {
			architectureResource.save(Collections.EMPTY_MAP);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static Resource loadInstance(URI uri) {
		ResourceSet rs = new ResourceSetImpl();
		Resource resource = rs.getResource(uri, true);
		return resource;
	}

	public static Activity getStoryDiagramActivityFromFile(String path)
			throws SDMException {
		Resource diagramResource = new ResourceSetImpl().getResource(
				URI.createFileURI(path), true);

		if (diagramResource.getContents().isEmpty()) {
			throw new SDMException(
					"Could not load story diagram at the given path: " + path);
		}

		Activity activity = (Activity) diagramResource.getContents().get(0);

		if (activity == null) {
			throw new SDMException(
					"Could not found activity in story diagram. Story diagram is invalid.");
		}

		return activity;
	}

	public static void initializeStoryDiagrams() {
		/*
		 * Register the mlsdm story diagram package.
		 */
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put(
				"mlsdm", new XMIResourceFactoryImpl());
		/*
		 * Initialize all related story diagram packages. Calling the eClass
		 * triggers the initialization.
		 */
		MlsdmPackage.eINSTANCE.eClass();
		MlexpressionsPackage.eINSTANCE.eClass();
		MlcallactionsPackage.eINSTANCE.eClass();
	}

	public static MLSDMInterpreter getStoryDiagramInterpreter() {
		/*
		 * Create the manager for expression languages.
		 */
		MLSDMExpressionInterpreterManager manager = new MLSDMExpressionInterpreterManager(
				BatchTest.class.getClassLoader());

		/*
		 * For debugging and development purpose, notifications can be activated
		 * and write to stdout.
		 */
		MLSDMOutputStreamNotificationReceiver notificationStreamWriter = new MLSDMOutputStreamNotificationReceiver();
		manager.getNotificationEmitter().addNotificationReceiver(
				notificationStreamWriter);

		return new MLSDMInterpreter(manager);
	}

}
