package mRUBiS_Tasks;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;

import de.mdelab.mlsdm.Activity;
import de.mdelab.mlsdm.interpreter.MLSDMInterpreter;
import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.CF1;
import de.mdelab.morisia.comparch.CF2;
import de.mdelab.morisia.comparch.CF3;
import de.mdelab.morisia.comparch.CF4;
import de.mdelab.morisia.comparch.CF5;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Capability;
import de.mdelab.morisia.comparch.simulator.ComparchSimulator;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.impl.testTrace;
import de.mdelab.morisia.comparch.simulator.impl.ValidationInjectionStrategy;
import de.mdelab.morisia.selfhealing.incremental.EventListener;
import de.mdelab.morisia.selfhealing.incremental.EventQueue;
import de.mdelab.morisia.selfhealing.rules.CostPredictor;
import de.mdelab.morisia.selfhealing.rules.IssueComparator;
import de.mdelab.morisia.selfhealing.rules.PerformanceEfficiencyManager;
import de.mdelab.morisia.selfhealing.rules.RuleSelector;
import de.mdelab.morisia.selfhealing.rules.StaticIssueComparator;
import de.mdelab.morisia.selfhealing.rules.UtilityIncreasePredictor;
import de.mdelab.sdm.interpreter.core.SDMException;
import de.mdelab.expressions.interpreter.core.variables.Variable;
import de.mdelab.mlsdm.interpreter.facade.OptimizedMSLDMInstanceFacade;
import de.mdelab.morisia.selfhealing.Approaches;
import de.mdelab.morisia.selfhealing.ArchitectureUtilCal;
import de.mdelab.morisia.selfhealing.EnvSetUp;
import de.mdelab.morisia.selfhealing.Utilityfunction;
import mRUBiS.Observations.Observations;

public class mRUBiS_Test {
	public static Approaches CURRENT_APPROACH = Approaches.Udriven;
	   //public static Approaches CURRENT_APPROACH = Approaches.Learning;
		public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Combined;
	
	
		public static FileWriter MLfile = null;
		public static FileWriter MLfileReP = null;
		public static FileWriter MLValidation = null;
		private final static int ROUNDS = 1; // 100; // 25
		private final static int RUNS = 5; // 1000

		private final static String SEP = ",";
		private final static boolean excel = true;
		private final static boolean Validation = false;
		private final static boolean optimal = true;
		
		
		public static void main(String[] args) throws SDMException, IOException {

			boolean enableLogging = false;
		

			/*
			 * Init model and SDM interpreter
			 */
			EnvSetUp.initialize();
			Activity A_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF1.mlsdm");
			Activity A_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF2.mlsdm");
			Activity A_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF3.mlsdm");
			Activity A_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF5_ALL.mlsdm");
			
			 if ( UTILITY_FUNCTION == Utilityfunction.Discontinuous){	// A_CF_New .....Discontinuous
				 A_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF1_New.mlsdm");
			 A_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF2_New.mlsdm");
				 A_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF3_New.mlsdm");
				 A_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF5_ALL.mlsdm");}
			
			else if( UTILITY_FUNCTION == Utilityfunction.Saturating){	// A_CF_Performance .....Saturating
				 A_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF1_Performance.mlsdm");
				 A_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF2_Performance.mlsdm");
				 A_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF3_Performance.mlsdm");
				 A_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF5_Performance.mlsdm");}
			
			else if( UTILITY_FUNCTION == Utilityfunction.Combined){	// A_CF_ALL ..... ALL + ALL without Random
				 A_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF1_ALL.mlsdm");
				 A_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF2_ALL.mlsdm");
				 A_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF3_ALL.mlsdm");
				 A_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF5_ALL.mlsdm");}

			Activity P_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/planning/P_CF1.mlsdm");
			Activity P_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/planning/P_CF2.mlsdm");
			Activity P_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/planning/P_CF3.mlsdm");
			Activity P_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/planning/P_CF5.mlsdm");

			Activity E_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/Execute/E_CF1.mlsdm");
			Activity E_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/Execute/E_CF2.mlsdm");
			Activity E_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/Execute/E_CF3.mlsdm");
			Activity E_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/Execute/E_CF5.mlsdm");
			// Comparator for sorting the CF for execution
			Comparator<Issue> issueComparator = null;
			if (CURRENT_APPROACH == Approaches.Udriven ) {
				issueComparator = new IssueComparator();

		

			}else if (CURRENT_APPROACH == Approaches.Learning) {
				issueComparator = new RLIssueComparator();
			}

			/*
			 * get SDM interpreter
			 */
			boolean stdout = false;
			final boolean useOptimization = true;
			MLSDMInterpreter interpreter = EnvSetUp.getStoryDiagramInterpreter(stdout, useOptimization);
			 // Benchmark actually starts
			 //
			for (int i = 0; i < ROUNDS; i++) {
				System.out.println("\n" + (i + 1) + ". round starts...");

				/*
				 * Load model
				 */
				Resource architectureResource = EnvSetUp
						// .loadFreshInstance("model/mRUBiS-100Saturation.comparch");
						.loadFreshInstance("model/enriched/mRUBiS-1shop_enriched.comparch");// ALL
																									// Final
																									// results
				// .loadFreshInstance("model/mRUBiS-100ML_adjustedReq.comparch");
				Architecture architecture = (Architecture) architectureResource.getContents().get(0);

				// EMF Delete Optimization
				if (useOptimization) {
					((OptimizedMSLDMInstanceFacade) interpreter.getFacadeFactory().getInstanceFacade())
							.initialize(Collections.singleton(architecture));
				}

				Annotations annotations = architecture.getAnnotations();
				if (annotations == null) {
					annotations = ComparchFactory.eINSTANCE.createAnnotations();
					architecture.setAnnotations(annotations);
				}
				if (excel) {
					System.out.println("Overall utility: " + ArchitectureUtilCal.computeOverallUtility(architecture));
					
					for(Tenant shop:architecture.getTenants())
					{
						for (Component component : shop.getComponents())
						{
						//System.out.println("\n Criticality of Type :  " +component.getType().getCriticality());
						
						
						}
					

				}
				
				
				}
				// attach event listener
				architecture.eAdapters().add(new EventListener());

				// set up simulator:
				String logFile = null;
				boolean logToConsole = false;
				ComparchSimulator simulator = ComparchSimulator.FACTORY.createSimulator(Capability.SELF_REPAIR,
						architecture, RUNS, Level.CONFIG, logFile, logToConsole);
				InjectionStrategy strategy = new testTrace
						(simulator.getSupportedIssueTypes(), architecture);
				simulator.setInjectionStrategy(strategy);
				int issueCount = simulator.validate();
				if (issueCount > 0) {
					System.out.println("Initial validaton. There are already issues in the model.");
				}
				else {System.out.println("Initial validaton. There are no issues in the model.");}
	
	
			}//ROUNDS
}
}