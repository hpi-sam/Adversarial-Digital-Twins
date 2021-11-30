package de.mdelab.morisia.selfhealing;

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
import de.mdelab.morisia.selfhealing.Utilityfunction;
import mRUBiS.Observations.Observations;

public class Evaluation_ML {

   public static Approaches CURRENT_APPROACH = Approaches.Udriven;
 //  public static Approaches CURRENT_APPROACH = Approaches.ML;
   //public static Approaches CURRENT_APPROACH = Approaches.WORST;
	//public static Approaches CURRENT_APPROACH = Approaches.SOLVER;
	// public static Approaches CURRENT_APPROACH = Approaches.STATIC;
	// public static Approaches CURRENT_APPROACH = Approaches.SEMI_STATIC;
	//public static Approaches CURRENT_APPROACH = Approaches.RANDOM;

  // public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Linear;
	//public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Discontinuous;
	//public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Saturating;
	public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Combined;
	
	public static FileWriter MLfile = null;
	public static FileWriter MLfileReP = null;
	public static FileWriter MLValidation = null;
	private final static int ROUNDS = 1; // 100; // 25
	private final static int RUNS = 5; // 1000

	private final static String SEP = ",";
	private final static boolean excel = false;
	private final static boolean Validation = false;
	private final static boolean optimal = true;
	
	
	public static void main(String[] args) throws SDMException, IOException {

		boolean enableLogging = false;
		configLogging(enableLogging);

		/*
		 * Init model and SDM interpreter
		 */
		EnvSetUp.initialize();

		/*
		 * load rules
		 */
		// A_CF .....Linear
		// A_CF_ADT ..... Probabilistic on ADT
		// A_CF_New .....Discontinuous
		// A_CF_Performance .....Saturating
		// A_CF_ALL ..... ALL + ALL without Random
		
		       
		
		//if( UTILITY_FUNCTION == Utilityfunction.Linear){	// A_CF .....Linear 
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
		if (CURRENT_APPROACH == Approaches.Udriven ||CURRENT_APPROACH == Approaches.ML ||CURRENT_APPROACH == Approaches.SEMI_STATIC) {
			issueComparator = new IssueComparator();

		} else if (CURRENT_APPROACH == Approaches.STATIC) {
			issueComparator = new StaticIssueComparator();
			
		}
		/*
		 * get SDM interpreter
		 */
		boolean stdout = false;
		final boolean useOptimization = true;
		MLSDMInterpreter interpreter = EnvSetUp.getStoryDiagramInterpreter(stdout, useOptimization);

		/*
		 * Measurements
		 */
		double st_A = 0.0;
		double sst_A = 0.0;
		double st_P = 0.0;
		double st_SORT = 0.0;
		double sst_P = 0.0;
		double st_E = 0.0;
		double sst_E = 0.0;
		// run rules on the real system!
		double st_Run = 0.0;
		double sst_Run = 0.0;
		int evalRound=3;
		
		
		
		
		double MTTR=2; //sec
		double MTBF=0; //sec
		double plantime=0; //sec
		double AccReward=0;
		double Reward=0;
        
		
		FileWriter utilityWriter = null;
		FileWriter timeWriter = null;
		// FileWriter fileWriter = null;

		
		if (excel) {
			
			
			utilityWriter = new FileWriter("measurements/ML/Reward/Reward.csv");
			utilityWriter.append("RUN" +  SEP + "Overal_Utility" + SEP
					 + "Utility_Drop" + SEP+ "Utility_Icrease" + SEP
					+ "# of CF"+SEP+"plan time (sec)"+SEP+"MTBF"+SEP+"Reward" +SEP+"ACC_Reward"+"\n");
			
			
			MLfile = new FileWriter("measurements/ML/test.csv");
			MLfileReP = new FileWriter("measurements/ML/testreq.csv");
			MLfile.append("FAILURE_NAME" + SEP + "UTILITY_DROP" + SEP + "AFFECTED_COMPONENT" + SEP + "CRITICALITY" + SEP
					+ "UTILITY_INCREASE" + SEP + "COSTS" + SEP + "CONNECTIVITY" + SEP + "RELIABILITY" + SEP
					+ "IMPORTANCE" + SEP + "PROVIDED_INTERFACE" + SEP + "REQUIRED_INTERFACE" + SEP + "ADT" + SEP
					+ "RULE" + SEP + " PMax" + SEP + "alpha" + SEP + "REPLICA" + SEP + "REQUEST" + SEP + "0" + SEP + "1"
					+ SEP + "FO" + "\n");

			MLfileReP.append("REPLICA" + SEP + "INCREASE" + SEP + "Reliability" + "\n");
		}
		if (Validation) {
			MLValidation = new FileWriter("measurements/ML/Ranking/9K/5-25-50/GBM/Ranking.csv");
			//MLValidation.append("Index" + SEP + "Failure" + SEP + "Affected_Component_Uid" + SEP + "Affected_Component"
			//		+ SEP + "Utility_Drop" + SEP + "Rule" + SEP + "Utility_Increase" + "\n");
		
	if (!optimal)
		MLValidation.append("Preditcted_Index"+ SEP +"Preditcted_Failure"+ SEP +"Preditcted_Affected_Component_Uid"+ SEP 
				+"	Preditcted_Affected_Component"+ SEP +"	Preditcted_Utility_Drop"+ SEP +"	Preditcted_Rule"+	SEP +"Preditcted_Utility_Increase"+ SEP +
				"CycleID"+ SEP +	"CycleSize"+"\n");
	else
		MLValidation.append("Optimal_Index "+ SEP +	"	Optimal_Failure"+ SEP +	"	Optimal_Affected_Component_Uid	"+ SEP +	
				"Optimal_Affected_Component"+ SEP +	"	Optimal_Utility_Drop"+ SEP +	"	Optimal_Rule	"+ SEP +	""
						+ "Optimal_Utility_Increase"+ SEP +"CycleID"+ SEP +	"CycleSize"+"\n");
		}	
	/*
		 * Benchmark actually starts
		 */
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
				System.out.println("Overall utility: " + computeOverallUtility(architecture));

			}

			// attach event listener
			architecture.eAdapters().add(new EventListener());

			// set up simulator:
			String logFile = null;
			boolean logToConsole = false;
			ComparchSimulator simulator = ComparchSimulator.FACTORY.createSimulator(Capability.SELF_REPAIR,
					architecture, RUNS, Level.CONFIG, logFile, logToConsole);
			//InjectionStrategy strategy = new DEUGCF5_BurstInjectionStrategy
			InjectionStrategy strategy = new testTrace
        	//InjectionStrategy strategy = new DEUG_BurstInjectionStrategy
			//InjectionStrategy strategy = new ValidationCF5InjectionStrategy
			//InjectionStrategy strategy = new ValidationInjectionStrategy
					//InjectionStrategy strategy = new FairInjectionStrategy
					(simulator.getSupportedIssueTypes(), architecture);
			simulator.setInjectionStrategy(strategy);

			long sum_nano_time_A = 0;
			long sum_nano_time_P = 0;
			long sum_nano_time_E = 0;
			long sum_nano_time_Run = 0;
			long sum_nano_time_SORT = 0;

			/*
			 * Start the simulation
			 */

			// call the simulator for the initial validation
			int issueCount = simulator.validate();
			if (issueCount > 0) {
				System.out.println("Initial validaton. There are already issues in the model.");
			}
			double UDrop=0;
			double UIncrease=0;
	  	    double	OveralU=computeOverallUtility(architecture);
			int run = 0;
			while (!simulator.isSimulationCompleted()) { // = number of RUNS
				run++;
				if (run % 100 == 0)
					System.out.println();
				System.out.print(".");

				// call the simulator to inject issues.
				simulator.injectIssues();
				

				   Observations.makeObservation(architecture);
				// if run <= RUNS then the simulator injects issues. As soon as
				// run > RUNS, the simulator is triggered only once to analyze
				// the model and the last adaptation.
				if (run <= RUNS) {

					/*
					 * Analyze
					 */
					long start_A = System.nanoTime();
					analyze(interpreter, annotations, A_CF1, A_CF2, A_CF3, A_CF5);
					sum_nano_time_A += System.nanoTime() - start_A;
					 System.out.printf("\n Analyze detects "+ annotations.getIssues().size()+" issues");
        
					/*
					 * Plan
					 */
					long start_P = System.nanoTime();
					plan(interpreter, annotations, P_CF1, P_CF2, P_CF3, P_CF5);
					// Sorting the failures to address first
					List<Issue> allIssues = new LinkedList<>();
					allIssues.addAll(annotations.getIssues());
					 System.out.printf("\n Plan detects "+ allIssues.size()+" issues");
					

			 if (CURRENT_APPROACH == Approaches.RANDOM) {
						shuffle(allIssues);
					} else {
						long start_sort = System.nanoTime();
						allIssues.sort(issueComparator);
						sum_nano_time_SORT += System.nanoTime() - start_sort;

						// System.out.printf("SORT ours: "+
						// sum_nano_time_SORT/1e6 + "ms");
					}
					sum_nano_time_P += System.nanoTime() - start_P;
					plantime=(System.nanoTime() - start_P)/ 1e9; // sec
					// log the reward
					if (CURRENT_APPROACH == Approaches.STATIC) {// ||
																// CURRENT_APPROACH
																// ==
																// Approaches.RANDOM)
																// {
						// the static approach does not predict any costs or
						// utility increase. For measurements, compute them now.
						for (Issue issue : allIssues) {
							UtilityIncreasePredictor.semiStaticPredictUtility(issue);
							CostPredictor.predictCosts(issue);
						}
					}
					
					// add Real Utility values
					if (CURRENT_APPROACH == Approaches.ML)
					{for (Issue issue : allIssues)
						RuleSelector.addActualUtilityIncreaseToRule( issue, UTILITY_FUNCTION);
					}
					st_SORT += sum_nano_time_SORT / 1e6;
					
					 if (excel)
		   				{// calculate the drop
						 UDrop=0;
						 UIncrease=0;
						 plantime=2; // uncommnet for 5-25-50
						 Reward=0;
						 int cfnumber=allIssues.size();
						 for (Issue issue : allIssues)
						 {	UDrop+=issue.getUtilityDrop() ;
						 UIncrease+=issue.getHandledBy().get(0).getUtilityIncrease(); }
						 OveralU=OveralU-UDrop;
						//Reward=OveralU*plantime; // commnet for 5-25-50      // running with droped utility for the planning period
						 utilityWriter.append("\n" + run
			   						+SEP + OveralU
			   						+SEP + UDrop
			   						+SEP + UIncrease
			   						+SEP + cfnumber
			   						+SEP+plantime
			   						+SEP+SEP+Reward
			   						//+SEP+AccReward // commnet for 5-25-50
			   						);
						 AccReward += Reward;
						// int period=allIssues.size()-1;
						 
						  Reward=0;
						 for (Issue issue : allIssues)
						 { OveralU+=issue.getHandledBy().get(0).getUtilityIncrease(); 
						// System.out.print("\n increase "+ issue.getHandledBy().get(0).getUtilityIncrease());
							 Reward+= OveralU*MTTR;
							 
						 }
						 
						MTBF= mtbfCalc(run+1);
						if(CURRENT_APPROACH == Approaches.ML)
					    MTBF=MTBF-plantime;
						// Reward+=OveralU*MTBF; // commnet for 5-25-50
						 AccReward +=Reward;
							 
						 utilityWriter.append("\n" + run
		   						+SEP + OveralU
		   						+SEP+SEP
		   						+SEP+cfnumber
		   						+SEP
		   						+SEP+MTBF
		   						+SEP+Reward
		   						+SEP+AccReward 
		   						);
		   				
		               }
					System.out.print("\n Run : " + run);
					
					if (excel) {
						//System.out.print("\n Run : " + run);
						for (Issue issue : allIssues) {
							Rule r = issue.getHandledBy().get(0);
							// ** Compute the Connectivity
							// first, init with number outgoing connectors
							int numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
							int numberOfProvided = 0;
							// second, add number of incoming connectors
							for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
								numberOfProvided = numberOfProvided + pi.getConnectors().size();
							}
							numberOfConnectors = numberOfConnectors + numberOfProvided;
							// System.out.print( "\n ... > "+ r.getUtilityIncrease() + SEP );
							// "+issue.getAffectedComponent().getType().getName()+"size
							// of req " + issue.getAffectedComponent()
							// .getRequiredInterfaces().size() + " size of prov:
							// " + numberOfProvided + " connectivity " +
							// numberOfConnectors);
							// ** Compute the Reliability
							double reliability = 1;
							boolean replace = false;
							if (r instanceof ReplaceComponent) {

								ComponentType altComponentType = UtilityIncreasePredictor
										.selectAlternativeComponentType(issue.getAffectedComponent());

								if (altComponentType != null) {
									reliability = altComponentType.getReliability();
									replace = true;
								//	System.out.print("\n reliability    " + reliability + "  rel"
									//		+ issue.getAffectedComponent().getType().getReliability());
								}

							} else {
								reliability = issue.getAffectedComponent().getType().getReliability();
							}

							double zero = issue.getAffectedComponent().getType().getCriticality() 
									* reliability
									* issue.getAffectedComponent().getTenant().getImportance()
									* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1)
									- 10 * issue.getAffectedComponent().getADT();
							zero = zero - r.getUtilityIncrease();

							double one = issue.getAffectedComponent().getType().getCriticality()
									* reliability
									* issue.getAffectedComponent().getTenant().getImportance()
									* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1) * 2
									- 10 * issue.getAffectedComponent().getADT();
							one = one - r.getUtilityIncrease();

							double FO = issue.getAffectedComponent().getType().getCriticality() 
									* reliability
									* issue.getAffectedComponent().getTenant().getImportance()
									* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1)
									* numberOfProvided - 10 * issue.getAffectedComponent().getADT();
							FO = FO - r.getUtilityIncrease();

							MLfile.append(issue.eClass().getName() + SEP 
									+ issue.getUtilityDrop() + SEP
									+ issue.getAffectedComponent().getType().getName() + SEP
									+ issue.getAffectedComponent().getType().getCriticality() + SEP
									+ r.getUtilityIncrease() + SEP 
									+ r.getCosts() + SEP
									+ numberOfConnectors + SEP
									+ reliability + SEP 
									+ issue.getAffectedComponent().getTenant().getImportance() + SEP
									+ numberOfProvided + SEP
									+ issue.getAffectedComponent().getRequiredInterfaces().size() + SEP
									+ issue.getAffectedComponent().getADT() + SEP 
									+ r.eClass().getName() + SEP
									+ issue.getAffectedComponent().getType().getPerformanceMax() + SEP
									+ (4 / issue.getAffectedComponent().getType().getSatPoint()) + SEP
									+ issue.getAffectedComponent().getInUseReplica() + SEP
									+ issue.getAffectedComponent().getRequest() + SEP 
									+ zero + SEP
									+ one + SEP
									+ FO
									+ "\n"

							);
							// System.out.print( "\n ... > "+ r.getUtilityIncrease() + "    " + r.eClass().getName() );
						/*	System.out.print("\n"+issue.eClass().getName()+ "rule "+ r.eClass().getName()+"  increase "+ r.getUtilityIncrease() + " cost"
									+ r.getCosts() + SEP);*/
						}
						 MLfile.append("\n");
					}
					if (Validation) {
						int index = 1;
						for (Issue issue : allIssues) {
							Rule r = issue.getHandledBy().get(0);
							MLValidation.append(index + SEP + issue.eClass().getName() + SEP
									+ issue.getAffectedComponent().getUid() + SEP
									+ issue.getAffectedComponent().getType().getName() + SEP + issue.getUtilityDrop()
									+ SEP + r.eClass().getName() + SEP + r.getUtilityIncrease() + SEP +run + "." + (evalRound )+SEP+allIssues.size()+ "\n");
							index++;
						}
					if(run==5||run==25||run==50||
								  run==56||run==76||run==101||
								  run==107||run==127||run==152||
								  run==158||run==178||run==203||
								  run==209||run==229||run==254||
								  run==260||run==280||run==305)
						if(run==50)
						MLValidation.append("\n");
					}

					/*
					 * Retrieve the predicted costs of the last remaining rule
					 * for each issue. The rules in the model are those that are
					 * executed next.
					 */
					List<Rule> rulesToBeExecuted = annotations.getRules();
					for (Rule r : rulesToBeExecuted) {
						sum_nano_time_Run += r.getCosts() * 1e9; // sec to ns
					}

					/*
					 * Execute
					 */
					List<Component> BACKUPcomp = new LinkedList<>();
					for (Issue issue : allIssues) {
						// System.out.print("\n here
						// "+issue.getAffectedComponent().getType());
						BACKUPcomp.add(issue.getAffectedComponent());
					}
					// for (Component component : BACKUPcomp)
					// {component.}

					long start_E = System.nanoTime();
					execute(interpreter, allIssues, E_CF1, E_CF2, E_CF3, E_CF5);
					annotations.getIssues().clear();
					annotations.getRules().clear();
					sum_nano_time_E += System.nanoTime() - start_E;
					for (Component component : BACKUPcomp) {
						// System.out.print("\n here "+component);
						// //omponent.getType().getReliability() ;
						int numberOfConnectors = component.getRequiredInterfaces().size();
						int numberOfProvided = 0;
						// second, add number of incoming connectors
						for (ProvidedInterface pi : component.getProvidedInterfaces()) {
							numberOfProvided = numberOfProvided + pi.getConnectors().size();
						}
					
    if (excel)
						MLfileReP.append(component.getInUseReplica() + SEP
						// +utilityIncrease+SEP
						// +component.getType().getReliability() +SEP
								+ "\n"

						);
					}
				}

				// call the simulator to validate the model.
				issueCount = simulator.validate();
				if (issueCount > 0) {
					System.out.println("\n Issue are remaining in the model!!!!! \n");
				}

			} // next simulation run

			double avg_nano_time_A = sum_nano_time_A / RUNS;  //sum_nano_time_ : is the sum of all runs
			st_A += avg_nano_time_A;
			sst_A += avg_nano_time_A * avg_nano_time_A;

			double avg_nano_time_P = sum_nano_time_P / RUNS;
			st_P += avg_nano_time_P;
			sst_P += avg_nano_time_P * avg_nano_time_P;

			double avg_nano_time_E = sum_nano_time_E / RUNS;
			st_E += avg_nano_time_E;
			sst_E += avg_nano_time_E * avg_nano_time_E;

			double avg_nano_time_Run = sum_nano_time_Run / RUNS;
			st_Run += avg_nano_time_Run;
			sst_Run += avg_nano_time_Run * avg_nano_time_Run;

			/*
			 * Save model
			 */
			// EnvSetUp.save(architectureResource, "model/mRUBiS_RESULT_" + i
			// + ".comparch");
			int pi = i + 1;
			System.out.println("\n\nRESULTS up to ROUND " + pi + " : ");
			String format = "%-10s %15.2f ns %15.2f ns %10.2f ms %10.2f ms %15.2f percent%n";

			double mean_A = st_A / pi;
			double sdev_A = Math.sqrt((sst_A + mean_A * mean_A * pi - 2 * mean_A * st_A) / (pi - 1));
			double per_A = (sdev_A / mean_A) * 100;
			System.out.printf(format, "ANALYZE", mean_A, sdev_A, mean_A / 1e6, sdev_A / 1e6, per_A);

			double mean_P = st_P / pi;
			double sdev_P = Math.sqrt((sst_P + mean_P * mean_P * pi - 2 * mean_P * st_P) / (pi - 1));
			double per_P = (sdev_P / mean_P) * 100;
			System.out.printf(format, "PLAN", mean_P, sdev_P, mean_P / 1e6, sdev_P / 1e6, per_P);

			double mean_E = st_E / pi;
			double sdev_E = Math.sqrt((sst_E + mean_E * mean_E * pi - 2 * mean_E * st_E) / (pi - 1));
			double per_E = (sdev_E / mean_E) * 100;
			System.out.printf(format, "EXECUTE", mean_E, sdev_E, mean_E / 1e6, sdev_E / 1e6, per_E);

			double mean_Run = st_Run / pi;
			double sdev_Run = Math.sqrt((sst_Run + mean_Run * mean_Run * pi - 2 * mean_Run * st_Run) / (pi - 1));
			double per_Run = (sdev_Run / mean_Run) * 100;
			System.out.printf(format, "RUN", mean_Run, sdev_Run, mean_Run / 1e6, sdev_Run / 1e6, per_Run);

			if (excel) {
				System.out.println("Overall utility!: " + computeOverallUtility(architecture));

			}
		} // next round

		System.out.println("\n\nRESULTS:");
		String format = "%-10s %15.2f ns %15.2f ns %10.2f ms %10.2f ms%n";

		double mean_A = st_A / ROUNDS;
		double sdev_A = Math.sqrt((sst_A - mean_A * mean_A * ROUNDS) / (ROUNDS - 1));
		System.out.printf(format, "ANALYZE", mean_A, sdev_A, mean_A / 1e6, sdev_A / 1e6);

		double mean_P = st_P / ROUNDS;
		double sdev_P = Math.sqrt((sst_P - mean_P * mean_P * ROUNDS) / (ROUNDS - 1));
		System.out.printf(format, "PLAN", mean_P, sdev_P, mean_P / 1e6, sdev_P / 1e6);

		double mean_E = st_E / ROUNDS;
		double sdev_E = Math.sqrt((sst_E - mean_E * mean_E * ROUNDS) / (ROUNDS - 1));
		System.out.printf(format, "EXECUTE", mean_E, sdev_E, mean_E / 1e6, sdev_E / 1e6);

		double mean_Run = st_Run / ROUNDS;
		double sdev_Run = Math.sqrt((sst_Run - mean_Run * mean_Run * ROUNDS) / (ROUNDS - 1));
		System.out.printf(format, "RUN", mean_Run, sdev_Run, mean_Run / 1e6, sdev_Run / 1e6);

		double mean_SORT = st_SORT / ROUNDS;
		// System.out.printf("SORT: "+ mean_SORT + "ms");

		if (excel) {
			MLfile.flush();
			MLfile.close();
			MLfileReP.flush();
			MLfileReP.close();
			utilityWriter.flush();
			utilityWriter.close();
		}
		if (Validation) {
			MLValidation.flush();
			MLValidation.close();
		}

	}
	private static double mtbfCalc(int run) {
//		int max=3;// STATIC: 2 OURS:3 SOLVER:7 
//		int min=1;
//		Random rnd = new Random();
//		  int MTBF = rnd.nextInt((max - min) + 1) + min;
//		System.out.print("\n\n MTBF   : "+MTBF);
//		return MTBF;
		
		
		Random rngIat = new Random(run);
		double MTBF=0;
		 // FOR DEUGE
		// parameters of the LogN distribution
		double meanIat = -2.28;       //-1.39; Grid5000
		double stddevIat= 1.35;      //1.03; Grid5000
		double[] IatValue = new double[RUNS+2];
		for (int i = 0; i <= (RUNS+1); ++i)
			
		{double stdNormalIat = rngIat.nextGaussian();
		double normalValueIat = stddevIat*stdNormalIat+ meanIat;
		IatValue[i] = Math.floor( Math.exp(normalValueIat)*3600);
		}
		MTBF=IatValue[run];
	
		
		//System.out.print("\n\n MTBF   : "+MTBF+"run " + run);
		
		return MTBF;
	}


	private static void shuffle(List<Issue> allIssues) {

		Collections.shuffle(allIssues);
		// shuffle the issues orders

	}

	private static double computeOverallUtility(Architecture architecture) {
		double overallUtility = 0;
		for (Tenant shop : architecture.getTenants()) {
			for (Component component : shop.getComponents()) {
				double reliability = component.getType().getReliability();
				double outgoingConnectors = component.getRequiredInterfaces().size();
				double incomingConnectors = 0;
				for (ProvidedInterface providedInterface : component.getProvidedInterfaces()) {
					incomingConnectors += providedInterface.getConnectors().size();
				}
				overallUtility += reliability * component.getType().getCriticality()
						* (outgoingConnectors + incomingConnectors);
			}
		}

		return overallUtility;
	}

	private static void analyze(MLSDMInterpreter interpreter, Annotations annotations, Activity A_CF1, Activity A_CF2,
			Activity A_CF3, Activity A_CF5) throws SDMException {
		while (!EventQueue.EVENTS.isEmpty()) {
			Notification notification = EventQueue.EVENTS.poll();

			Object notifier = notification.getNotifier();
			Object feature = notification.getFeature();
			// Change of the Component state --> check for CF1 or CF3
			if (feature == ComparchPackage.Literals.COMPONENT__STATE) {
				Component component = (Component) notifier;
				// System.out.print("\n Component "+component+"\n");

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("component", component));
				parameters.add(createParameter("annotations", annotations));

				// execute A_CF1
				interpreter.executeActivity(A_CF1, parameters);
				// execute A_CF3
				interpreter.executeActivity(A_CF3, parameters);

			} else
			// Change of the number of Failures attached to a ProvidedInterface
			// --> check for CF2
			if (feature == ComparchPackage.Literals.PROVIDED_INTERFACE__FAILURES) {
				ProvidedInterface providedInterface = (ProvidedInterface) notifier;
				// System.out.print("\n Component cf3
				// "+providedInterface.getComponent().getType().getName()+"\n");

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("providedInterface", providedInterface));
				parameters.add(createParameter("annotations", annotations));

				// execute A_CF2
				interpreter.executeActivity(A_CF2, parameters);

			} else
			// Add/Remove of a Connector from to a RequiredInterface --> check
			// for CF4
			if (feature == ComparchPackage.Literals.REQUIRED_INTERFACE__CONNECTOR) {
				// skip for now
			} else
			// Drop of the Efficiency of a component --> check
			// for CF5
			// if (feature ==
			// ComparchPackage.Literals.ARCHITECTURAL_ELEMENT__EFFICIENCY) {
			if (feature == ComparchPackage.Literals.ARCHITECTURAL_ELEMENT__REQUEST) {
				Component component = (Component) notifier;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("component", component));
				parameters.add(createParameter("annotations", annotations));

				// execute A_CF5
				interpreter.executeActivity(A_CF5, parameters);

			}

		}

	}

	private static void plan(MLSDMInterpreter interpreter, Annotations annotations, Activity P_CF1, Activity P_CF2,
			Activity P_CF3, Activity P_CF5) throws SDMException {
		for (Issue issue : annotations.getIssues()) {
			System.out.print("\n name of the comp." +issue.getAffectedComponent().getType().getName()+"\n");
			// Get all the applicable Rules

			// System.out.print("\n Next Issue");

			if (issue instanceof CF1) {
				CF1 cf1 = (CF1) issue;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf1", cf1));
				parameters.add(createParameter("annotations", annotations));

				interpreter.executeActivity(P_CF1, parameters);
				 System.out.print("\n Plan detected CF1");
			} else if (issue instanceof CF2) {
				CF2 cf2 = (CF2) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf2", cf2));
				parameters.add(createParameter("annotations", annotations));

				interpreter.executeActivity(P_CF2, parameters);
				 System.out.print("\n Plan detected CF2");
			} else if (issue instanceof CF3) {
				CF3 cf3 = (CF3) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf3", cf3));
				parameters.add(createParameter("annotations", annotations));
				 System.out.print("\n Plan detected CF3");

				interpreter.executeActivity(P_CF3, parameters);
			} else if (issue instanceof CF4) {
				// skip for now
			} else if (issue instanceof CF5) {
				 System.out.print("\n Plan detected CF5");
				CF5 cf5 = (CF5) issue;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf5", cf5));
				parameters.add(createParameter("annotations", annotations));

				interpreter.executeActivity(P_CF5, parameters);

			}
		}
	}

	private static void execute(MLSDMInterpreter interpreter, List<Issue> allIssues, Activity E_CF1, Activity E_CF2,
			Activity E_CF3, Activity E_CF5) throws SDMException {
		/*
		 * EXECUTE
		 */
		for (int i = 0; i < allIssues.size(); i++) {
			Issue issue = allIssues.get(i);
			if (issue instanceof CF1) {
				CF1 cf1 = (CF1) issue;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf1", cf1));
				parameters.add(createParameter("rule", cf1.getHandledBy().get(0)));

				interpreter.executeActivity(E_CF1, parameters);
			} else if (issue instanceof CF2) {
				CF2 cf2 = (CF2) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf2", cf2));
				parameters.add(createParameter("rule", cf2.getHandledBy().get(0)));

				interpreter.executeActivity(E_CF2, parameters);
			} else if (issue instanceof CF3) {
				CF3 cf3 = (CF3) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf3", cf3));
				parameters.add(createParameter("rule", cf3.getHandledBy().get(0)));

				interpreter.executeActivity(E_CF3, parameters);
			} else if (issue instanceof CF4) {
				// skip for now
			} else if (issue instanceof CF5) {

				CF5 cf5 = (CF5) issue;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf5", cf5));
				parameters.add(createParameter("rule", cf5.getHandledBy().get(0)));
				interpreter.executeActivity(E_CF5, parameters);
			}
		}

	}

	private static Variable<EClassifier> createParameter(String name, EObject value) {
		return new Variable<EClassifier>(name, value.eClass(), value);
	}

	// CONFIG LOGGING

	private static void configLogging(boolean enable) {
		Logger logger = Logger.getLogger("");
		logger.setLevel(Level.CONFIG);
		// suppress the logging output to the console
		removeConsoleHandler(logger);
		if (enable) {
			try {
				FileHandler fileTxt = new FileHandler("eval.log");
				// create a TXT formatter
				Formatter formatterTxt = new SimpleFormatter();
				formatterTxt = new java.util.logging.Formatter() {
					@Override
					public String format(LogRecord record) {
						return record.getLevel() + " " + record.getMessage() + "\n";
					}
				};
				fileTxt.setFormatter(formatterTxt);
				logger.addHandler(fileTxt);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	private static void removeConsoleHandler(Logger logger) {
		ConsoleHandler toBeRemoved = null;
		Handler[] handlers = logger.getHandlers();
		for (int i = 0; i < handlers.length; i++) {
			if (handlers[i] instanceof ConsoleHandler) {
				toBeRemoved = (ConsoleHandler) handlers[i];
			}
		}
		logger.removeHandler(toBeRemoved);
	}

	// END OF CONFIG LOGGING

}
