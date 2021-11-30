package de.mdelab.morisia.selfhealing;

import java.io.File;
import java.io.IOException;
import java.util.Collections;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import de.mdelab.mlcallactions.MlcallactionsPackage;
import de.mdelab.mlexpressions.MLExpression;
import de.mdelab.mlexpressions.MlexpressionsPackage;
import de.mdelab.mlexpressions.mlcallactions.CallActionsExpressionLanguageConstants;
import de.mdelab.mlsdm.Activity;
import de.mdelab.mlsdm.MlsdmPackage;
import de.mdelab.mlsdm.interpreter.MLSDMInterpreter;
import de.mdelab.mlsdm.interpreter.MLSDMCallActionInterpreter;
import de.mdelab.mlsdm.interpreter.MLSDMExpressionInterpreterManager;
import de.mdelab.mlsdm.interpreter.facade.OptimizedMLSDMMetamodelFacadeFactory;
import de.mdelab.mlsdm.interpreter.notifications.MLSDMOutputStreamNotificationReceiver;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.sdm.interpreter.core.SDMException;
import de.mdelab.sdm.interpreter.core.executionTrace.ExecutionTracePackage;
import de.mdelab.expressions.interpreter.ocl.OCLExpressionInterpreter;
import de.mdelab.expressions.interpreter.ocl.OCLExpressionInterpreterConstants;

public class EnvSetUp {

	private final static URI BASE = URI.createFileURI(new File(".")
			.getAbsolutePath());
	private static ResourceSet RESOURCE_SET = new ResourceSetImpl();

	public static void initialize() {
		// init comparch
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put(
				"comparch", new XMIResourceFactoryImpl());
		ComparchPackage.eINSTANCE.eClass();
		/*
		 * Working with the interpreter requires at first the registration of
		 * the story diagram resources.
		 */
		initializeStoryDiagrams();
	}

	public static Resource loadInstance(String relativeUri) {
		URI uri = URI.createFileURI(relativeUri).resolve(BASE);
		Resource resource = RESOURCE_SET.getResource(uri, true);
		return resource;
	}

	public static Resource loadFreshInstance(String relativeUri) {
		URI uri = URI.createFileURI(relativeUri).resolve(BASE);
		Resource resource = new ResourceSetImpl().getResource(uri, true);
		return resource;
	}

	public static Activity getStoryDiagramActivityFromFile(String path)
			throws SDMException {
		Resource diagramResource = RESOURCE_SET.getResource(
				URI.createFileURI(path).resolve(BASE), true);

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

	private static void initializeStoryDiagrams() {
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
		ExecutionTracePackage.eINSTANCE.eClass();
		Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put( "executiontrace", new XMIResourceFactoryImpl() );
	}
	
	public static MLSDMInterpreter getStoryDiagramInterpreter( boolean stdout )
	throws SDMException {
		return getStoryDiagramInterpreter( stdout, false );
	}

	public static MLSDMInterpreter getStoryDiagramInterpreter( 	boolean stdout,
																final boolean useOptimization )
					throws SDMException {
		/*
		 * Create the manager for expression languages.
		 */
		MLSDMExpressionInterpreterManager manager = new MLSDMExpressionInterpreterManager(
				EnvSetUp.class.getClassLoader());

		/*
		 * For debugging and development purpose, notifications can be activated
		 * and write to stdout.
		 */
		if (stdout) {
			MLSDMOutputStreamNotificationReceiver notificationStreamWriter = new MLSDMOutputStreamNotificationReceiver();
			manager.getNotificationEmitter().addNotificationReceiver(
					notificationStreamWriter);
		}

		final MLSDMInterpreter interpreter;

		// EMF Delete Optimization
		if ( useOptimization ) {
			final OptimizedMLSDMMetamodelFacadeFactory facadeFactory = new OptimizedMLSDMMetamodelFacadeFactory( Collections.singleton( ComparchPackage.eINSTANCE ) );
			interpreter = new MLSDMInterpreter( manager, facadeFactory );
		}
		else {
			interpreter = new MLSDMInterpreter( manager );
		}

		//  final MLSDMExecutionTraceNotificationReceiver trace = new MLSDMExecutionTraceNotificationReceiver();
		//  interpreter.getNotificationEmitter().addNotificationReceiver( trace.g );
		attachLanguages(interpreter);

		return interpreter;
	}

	private static void attachLanguages(MLSDMInterpreter interpreter)
			throws SDMException {
		/*
		 * Register as many languages as you want
		 */
		//  try {
		interpreter.getExpressionInterpreterManager()
		.registerExpressionInterpreter(
				new OCLExpressionInterpreter<MLExpression>(),
				OCLExpressionInterpreterConstants.OCL_LANGUAGE_NAME, OCLExpressionInterpreterConstants.OCL_LANGUAGE_VERSION);

		// EMF Delete Optimization
		final MLSDMCallActionInterpreter actionInterpreter;

		if ( interpreter.getFacadeFactory() instanceof OptimizedMLSDMMetamodelFacadeFactory ) {
			actionInterpreter = new MLSDMCallActionInterpreter() {

				@Override
				protected MLSDMInterpreter createMLSDMInterpreter() {
					return new MLSDMInterpreter( interpreter.getExpressionInterpreterManager(), interpreter.getFacadeFactory() );
				}
			};
		}
		else {
			actionInterpreter = new MLSDMCallActionInterpreter();
		}

		interpreter.getExpressionInterpreterManager().registerExpressionInterpreter( actionInterpreter,
				CallActionsExpressionLanguageConstants.CALL_ACTION_EXPRESSION_LANGUAGE_NAME,
				CallActionsExpressionLanguageConstants.CALL_ACTION_EXPRESSION_LANGUAGE_VERSION );
		//  } catch (SDMException e) {
		//   e.printStackTrace();
		//   throw e;
		//  }
	}
	public static void save(Resource resource, String relativeUri) {
		URI resultURI = URI.createFileURI(relativeUri).resolve(BASE);
		resource.setURI(resultURI);
		try {
			resource.save(Collections.EMPTY_MAP);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
