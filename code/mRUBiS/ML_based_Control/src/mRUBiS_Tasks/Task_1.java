package mRUBiS_Tasks;

import java.io.FileWriter;
import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

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
import de.mdelab.morisia.comparch.simulator.impl.Trace_1;
import de.mdelab.morisia.comparch.simulator.impl.Trace_2;
import de.mdelab.morisia.comparch.simulator.impl.Trace_Deterministic;
import de.mdelab.morisia.comparch.simulator.impl.testTrace;
import de.mdelab.morisia.selfhealing.incremental.EventListener;
import de.mdelab.morisia.selfhealing.incremental.EventQueue;
import de.mdelab.morisia.selfhealing.rules.CostPredictor;
import de.mdelab.morisia.selfhealing.rules.IssueComparator;
import de.mdelab.morisia.selfhealing.rules.PerformanceEfficiencyManager;
import de.mdelab.morisia.selfhealing.rules.RuleSelector;
import de.mdelab.morisia.selfhealing.rules.UtilityIncreasePredictor;
import de.mdelab.sdm.interpreter.core.SDMException;
import de.mdelab.expressions.interpreter.core.variables.Variable;
import de.mdelab.mlsdm.interpreter.facade.OptimizedMSLDMInstanceFacade;
import de.mdelab.morisia.selfhealing.Approaches;
import de.mdelab.morisia.selfhealing.ArchitectureUtilCal;
import de.mdelab.morisia.selfhealing.EnvSetUp;
import de.mdelab.morisia.selfhealing.Utilityfunction;
import mRUBiS.Observations.Observations;


public class Task_1 {

	//public static Approaches CURRENT_APPROACH = Approaches.Udriven;
	//public static Approaches CURRENT_APPROACH = Approaches.RANDOM;
	public static Approaches CURRENT_APPROACH = Approaches.Learning;
	public static Utilityfunction UTILITY_FUNCTION = Utilityfunction.Combined;


	public static FileWriter Training = null;
	public static FileWriter MLValidation = null;

	private static Path issueToRulesPath = Paths.get("issueToRulesMap.json");
	private static Path rulesToExecutePath= Paths.get("rulesToExecute.json");


	private final static int RUNS = 10000; 

	private final static String SEP = ",";
	private final static boolean Log = true;



	public static void main(String[] args) throws SDMException, IOException, InterruptedException {

		try {
			if (Files.exists(issueToRulesPath)) {
				Files.delete(issueToRulesPath);
			}
			if (Files.exists(rulesToExecutePath)) {
				Files.delete(rulesToExecutePath);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}


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





		Activity A_CF1 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF1_ALL.mlsdm");
		Activity A_CF2 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF2_ALL.mlsdm");
		Activity A_CF3 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF3_ALL.mlsdm");
		Activity A_CF5 = EnvSetUp.getStoryDiagramActivityFromFile("model/analysis/A_CF5_ALL.mlsdm");

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



		}
		//option for Ranking
		/*else if (CURRENT_APPROACH == Approaches.Learning) {
			issueComparator = new RLIssueComparator();
		}*/

		/*
		 * get SDM interpreter
		 */
		boolean stdout = false;
		final boolean useOptimization = true;
		MLSDMInterpreter interpreter = EnvSetUp.getStoryDiagramInterpreter(stdout, useOptimization);







		if (Log) {



			Training=new FileWriter("Logs/TrainingmRUBiS.csv");


			Training.append( "Shop"+SEP + "AFFECTED_COMPONENT"+ SEP+"FAILURE_NAME" + SEP + "CURRENT_UTILITY" + SEP + "OPTIMAL_UTILITY" + SEP+ "CRITICALITY" + SEP
					+ "COSTS" + SEP + "CONNECTIVITY" + SEP + "RELIABILITY" + SEP
					+ "IMPORTANCE" + SEP + "PROVIDED_INTERFACE" + SEP + "REQUIRED_INTERFACE" + SEP + "ADT" + SEP
					+ "RULE" + SEP + " PMax" + SEP + "alpha" + SEP + "In Use REPLICA" + SEP + "LOAD"+"\n");

		}

		/*
		 * Benchmark actually starts
		 */


		/*
		 * Load model
		 */
		Resource architectureResource = EnvSetUp

				.loadFreshInstance("model/enriched/mRUBiS-10shop_enriched.comparch");



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


		// attach event listener
		architecture.eAdapters().add(new EventListener());

		// set up simulator:
		String logFile = null;
		boolean logToConsole = false;
		ComparchSimulator simulator = ComparchSimulator.FACTORY.createSimulator(Capability.SELF_REPAIR,
				architecture, RUNS, Level.CONFIG, logFile, logToConsole);
		//InjectionStrategy strategy = new testTrace
		InjectionStrategy strategy = new Trace_Deterministic
				(simulator.getSupportedIssueTypes(), architecture);
		simulator.setInjectionStrategy(strategy);


		/*
		 * Start the simulation
		 */


		// call the simulator for the initial validation
		int issueCount = simulator.validate();
		double	OveralU=ArchitectureUtilCal.computeOverallUtility(architecture);
		int run = 0;
		if (issueCount > 0) {
			System.out.println("\n\nInitial validaton. There are already issues in the model.");

		}
		else {System.out.println("\n\nInitial validaton. There are no issues in the model.\n \n \nCurrent Overal Utility is: "+ OveralU);}


		while (!simulator.isSimulationCompleted()) { // = number of RUNS
			run++;

			// call the simulator to inject issues.
			simulator.injectIssues();


			// if run <= RUNS then the simulator injects issues. As soon as
			// run > RUNS, the simulator is triggered only once to analyze
			// the model and the last adaptation.
			if (run <= RUNS) {
				System.out.print("\n Run : " + run);
				System.out.print("\n . \n .");


				/*
				 * Analyze
				 */
				analyze(interpreter, annotations, A_CF1, A_CF2, A_CF3, A_CF5);
				System.out.printf("\n>> Analyze Compelete\n\n");
				ArchitectureUtilCal.computeOverallUtility(architecture);


				/*
				 * Plan
				 */
				plan(interpreter, annotations, P_CF1, P_CF2, P_CF3, P_CF5);
				// Sorting the failures to address first
				List<Issue> allIssues = new LinkedList<>();
				allIssues.addAll(annotations.getIssues());



				//					if (CURRENT_APPROACH == Approaches.RANDOM) 
				//						{shuffle(allIssues);}
				//					
				//						
				//						else //Udriven and Learning
				//							{
				//							allIssues.sort(issueComparator);
				//						}




				// add Real Utility Values : Ground Truth
				if (CURRENT_APPROACH == Approaches.Learning)
				{for (Issue issue : allIssues)
					RuleSelector.addActualUtilityIncreaseToRule( issue, UTILITY_FUNCTION);
				}





				if (Log) {
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

						double reliability = 1;
						if (r instanceof ReplaceComponent) {

							ComponentType altComponentType = UtilityIncreasePredictor
									.selectAlternativeComponentType(issue.getAffectedComponent());

							if (altComponentType != null) {
								reliability = altComponentType.getReliability();

							}

						} else {
							reliability = issue.getAffectedComponent().getType().getReliability();
						}


						Training.append( issue.getAffectedComponent().getTenant().getName()+SEP
								+ issue.getAffectedComponent().getType().getName() + SEP
								+ issue.eClass().getName() + SEP 
								+  ArchitectureUtilCal.computeComponentUtility(issue.getAffectedComponent())+SEP
								+ (ArchitectureUtilCal.computeComponentUtility(issue.getAffectedComponent())+r.getUtilityIncrease() )+ SEP 
								+ issue.getAffectedComponent().getType().getCriticality() + SEP
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
								+ "\n"

								);



					}

					Training.append("\n");
				}


				/*
				 * Retrieve the predicted costs of the last remaining rule
				 * for each issue. The rules in the model are those that are
				 * executed next.
				 */


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


				if (CURRENT_APPROACH == Approaches.Learning) {

					// Get custom fix ordering from the controller
					System.out.println("Waiting for Python to send order in which to apply fixes...");
					String fromPython = "";
					HashMap<String, HashMap<String, String>> fixOrder = new HashMap<String, HashMap<String, String>>();
					while(true) {
						fromPython = RuleSelector.in.readLine();

						try {
							// get order in which to apply fixes from python
							fixOrder = new ObjectMapper().readValue(fromPython, HashMap.class);

							RuleSelector.out.println("fix_order_received");
							RuleSelector.logger.println("fix_order_received");
							break;
						} catch (IOException e) {
							System.out.println("Did not receive valid json from Python:");
							System.out.println(fromPython);
							continue;
						}

					}

					// reorder the issues according to the order sent by the controller
					
					System.out.println("org: " + allIssues);
					
					List<Issue> orderedIssues = new LinkedList<>();
					for (String priority: fixOrder.keySet()) {
						String shopName = fixOrder.get(priority).get("shop");
						String issueName = fixOrder.get(priority).get("issue");
						String componentName = fixOrder.get(priority).get("component");

						for (Issue issue: allIssues) {
							if (issue.getAffectedComponent().getTenant().getName().equals(shopName)) {
								if (issue.getClass().getSimpleName().replaceAll("Impl", "").equals(issueName)) {
									if (issue.getAffectedComponent().getType().getName().equals(componentName)) {
										orderedIssues.add(issue);
									}
								}
							}
						}
					}

					System.out.println("ord: " + orderedIssues);
					
					List<Issue> issueDiff = new ArrayList(allIssues);
					issueDiff.removeAll(orderedIssues);
					
					if (issueDiff.size() > 0) {
						System.out.println("!!!Found unmatched issue: " + issueDiff.get(0).getClass().getSimpleName());
						System.out.println("Unmatched issue comp: " + issueDiff.get(0).getAffectedComponent().getType().getName());
						System.out.println("Unmatched issue shop: " + issueDiff.get(0).getAffectedComponent().getTenant().getName());
						System.out.println("Unmatched issue order: " + fixOrder);
					}
					
					System.out.println("Org and ord issues diff: " + issueDiff);

					assert orderedIssues.size() == allIssues.size();

					allIssues = orderedIssues;

				}

				execute(interpreter, allIssues, E_CF1, E_CF2, E_CF3, E_CF5);


				if (CURRENT_APPROACH == Approaches.Learning) {

					// sample affected components one more time (get all params sampled in getComponentsUtility)
					ObjectMapper mapper = new ObjectMapper();
					HashMap<String, HashMap<String, HashMap<String, Double>>> issueToRulesMapFromFile = null;
					try {
						issueToRulesMapFromFile = mapper.readValue(issueToRulesPath.toFile(), HashMap.class);
					} catch (IOException e2) {
						e2.printStackTrace();
					}


					System.out.println("Waiting for Python to send fixed components JSON...");
					String fromPython = "";
					while(true) {
						fromPython = RuleSelector.in.readLine();

						try {
							// get components fixed in this run from python
							//HashMap<String, List<String>> fixedComponents = new HashMap<String, List<String>>();
							HashMap<String, List<String>> fixedComponents = new HashMap<String, List<String>>();
							fixedComponents = new ObjectMapper().readValue(fromPython, HashMap.class);

							// lookup their current state and send to python
							String state = "not_available";
							state = Observations.getFixedComponentStatus(architecture, fixedComponents);
							RuleSelector.out.println(state);
							RuleSelector.logger.println(state);
							break;
						} catch (IOException e) {
							System.out.println("Did not receive valid json from Python:");
							System.out.println(fromPython);
							continue;
						}

					}


					annotations.getIssues().clear();
					annotations.getRules().clear();

					try {
						if (Files.exists(issueToRulesPath)) {
							Files.delete(issueToRulesPath);
						}
						if (Files.exists(rulesToExecutePath)) {
							Files.delete(rulesToExecutePath);
						}
					} catch (IOException e) {
						e.printStackTrace();
					}

				}

			}//nex Run

			// call the simulator to validate the model.
			issueCount = simulator.validate();
			if (issueCount > 0) {
				System.out.println("\n Issue are remaining in the model!!!!! \n");
			}
			System.out.printf("\n >>Execution Completed! \n\n");
			/*for (Tenant shop : architecture.getTenants()) {
					for (Component component : shop.getComponents()) {
						System.out.printf("\n\n +++ Component "+component.getType().getName());
						System.out.printf("\nCriticality "+ component.getType().getCriticality()); 
						System.out.printf("\nPerformance "+ component.getPerformance());
						System.out.printf("\nReliability "+component.getType().getReliability() );
						System.out.printf("\nImportance "+component.getTenant().getImportance());
						System.out.printf("\nADT"+component.getADT());}



				}*/
			System.out.println("\n Overall Utility After Execution: " + ArchitectureUtilCal.computeOverallUtility(architecture));

		} // next simulation run


		/*
		 * Save model
		 */
		// EnvSetUp.save(architectureResource, "model/mRUBiS_RESULT_" + i
		// + ".comparch");







		if (Log) {

			Training.flush();
			Training.close();

		}


	}



	private static void shuffle(List<Issue> allIssues) {

		Collections.shuffle(allIssues);
		// shuffle the issues orders

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
			System.out.print("\n Affected component: " +issue.getAffectedComponent().getType().getName()+"\n");

			// System.out.print("\n Next Issue");

			if (issue instanceof CF1) {
				CF1 cf1 = (CF1) issue;
				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf1", cf1));
				parameters.add(createParameter("annotations", annotations));

				interpreter.executeActivity(P_CF1, parameters);
				// System.out.print("\n Plan detected CF1");
			} else if (issue instanceof CF2) {
				CF2 cf2 = (CF2) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf2", cf2));
				parameters.add(createParameter("annotations", annotations));

				interpreter.executeActivity(P_CF2, parameters);
				// System.out.print("\n Plan detected CF2");
			} else if (issue instanceof CF3) {
				CF3 cf3 = (CF3) issue;

				Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
				parameters.add(createParameter("cf3", cf3));
				parameters.add(createParameter("annotations", annotations));
				// System.out.print("\n Plan detected CF3");

				interpreter.executeActivity(P_CF3, parameters);
			} else if (issue instanceof CF4) {
				// skip for now
			} else if (issue instanceof CF5) {
				// System.out.print("\n Plan detected CF5");
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


}
