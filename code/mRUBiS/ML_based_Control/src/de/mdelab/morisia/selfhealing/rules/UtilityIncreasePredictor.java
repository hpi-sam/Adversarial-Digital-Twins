package de.mdelab.morisia.selfhealing.rules;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import java.util.HashMap;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.mdelab.morisia.comparch.AddReplica;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.HwRedeployComponent;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.LwRedeployComponent;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RemoveReplica;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.RestartComponent;
import de.mdelab.morisia.comparch.Rule;



public class UtilityIncreasePredictor {

	private final static Logger LOGGER = Logger.getLogger(UtilityIncreasePredictor.class.getName());
	static long sum_nano_time_P = 0;


	/*
	 * OUR APPRAOCH - Calculating Combined Utility Increase
	 * ===============================================================
	 */
	/**
	 * Calculates the combined utility increase of each rule potentially
	 * handling the given issue.
	 * 
	 * @param issue
	 *            the given issue.
	 */
	public static void calculateCombinedUtilityIncrease(Issue issue) {

		LOGGER.info("\n --->Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		//System.out.print("\n Predicting utility increase for the " + issue.getHandledBy().size()
			//	+ " rules attached to the issue " );
		CostPredictor.predictCosts(issue);
		for (Rule rule : issue.getHandledBy()) {
			// Each rule increases the utility as much as the issue has
			// decreased it, except of the ReplaceComponent rule that may
			// further increase the utility when selecting a component of a
			// "better" alternative type.
			double utilityIncrease = 0;
			if (rule instanceof ReplaceComponent) {
				// select type of the alternative component;
				ComponentType altComponentType = selectAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors
					int numberOfConnectors = affectedComponent.getRequiredInterfaces().size();
					// second, add number of incoming connectors
					for (ProvidedInterface pi : affectedComponent.getProvidedInterfaces()) {
						numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
					}
					long start_P = System.nanoTime();

					utilityIncrease = altComponentType.getReliability() * affectedComponent.getType().getCriticality()
							* affectedComponent.getPerformance()
							* issue.getAffectedComponent().getTenant().getImportance()
							* (affectedComponent.getRequiredInterfaces().size() + 1);

					int numberOfProvided = 0;
					for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
						numberOfProvided = numberOfProvided + pi.getConnectors().size();
					}

					if (numberOfProvided > 1) {
						utilityIncrease = utilityIncrease * numberOfProvided
								- 10 * issue.getAffectedComponent().getADT();
					}

					else if (numberOfProvided == 1) {
						utilityIncrease = utilityIncrease * numberOfProvided * 2
								- 10 * issue.getAffectedComponent().getADT();
					}

					else {
						utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
					} // if(numberOfProvided ==0) utility increase does not
						// change

					sum_nano_time_P += System.nanoTime() - start_P;
					//System.out.print("\n   C utility calculation time   " + sum_nano_time_P + "ns\n");
				}
			} else if (rule instanceof RemoveReplica) {

				long start_P = System.nanoTime();

				double replica = 1;
				double newPerformance = 0;
				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));
				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				if (replica > 0) {
					replica = originalReplica - 1;
					newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
							* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
									* (replica / issue.getAffectedComponent().getRequest()));

					if (0.05 < ((oldPerformance - newPerformance) / oldPerformance))
						while (0.05 < (oldPerformance - newPerformance) && (replica > 0)) { // System.out.print("\n
																							// stuck
																							// here
																							// *");

							{
								replica--;
								oldPerformance = newPerformance;
								newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
										* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
												* (replica / issue.getAffectedComponent().getRequest()));

							}
						} // end while
				} // end if
				utilityIncrease = issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality() * newPerformance
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);

				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}
				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

				sum_nano_time_P += System.nanoTime() - start_P;
				//System.out.print("\n  C utility calculation time   " + sum_nano_time_P + "ns\n");
			} else if (rule instanceof AddReplica) {
				long start_P = System.nanoTime();

				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				double replica = originalReplica + 1;

				double newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
						* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
								* (replica / issue.getAffectedComponent().getRequest()));

				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));

				// System.out.print("\n stuck here2 ");
				if (0.01 < ((newPerformance - oldPerformance) / newPerformance))
					while (0.01 < ((newPerformance - oldPerformance) / newPerformance)) { // System.out.print("\n
																							// stuck
																							// here
																							// 3");
						replica++;
						// System.out.print("\n stuck here4 ");
						oldPerformance = newPerformance;
						newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
								* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
										* (replica / issue.getAffectedComponent().getRequest()));

					} // end while

				// System.out.print("\n AD Replica " + replica);
				utilityIncrease = issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality() * newPerformance
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);

				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}

				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

				sum_nano_time_P += System.nanoTime() - start_P;
				//System.out.print("\n   C utility calculation time   " + sum_nano_time_P + "ns\n");
			} else {
				long start_P = System.nanoTime();

				utilityIncrease = issue.getUtilityDrop();

				sum_nano_time_P += System.nanoTime() - start_P;
				//System.out.print("\n   C utility calculation time   " + sum_nano_time_P + "ns\n");
			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
			//System.out.print("\n Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
			
		}
		
		Path jsonFile = Paths.get("issueToRulesMap.json");
		String affectedComponent = issue.getAffectedComponent().getType().getName();
		String issueName = issue.getClass().getSimpleName().replaceAll("Impl", "");
		String shopName = issue.getAffectedComponent().getTenant().getName();
		List<Rule> availableRules = issue.getHandledBy();
		
		// store rules in the following way: shop -> issue -> component type -> rules, costs 

		HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> shopToissueToCompToRulesMap = new HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>>();
		HashMap<String, HashMap<String, HashMap<String, Double>>> issueToCompToRulesMap = new HashMap<String, HashMap<String, HashMap<String, Double>>>();  
		HashMap<String, HashMap<String, Double>> compToRulesMap = new HashMap<String, HashMap<String, Double>>();
		HashMap<String, Double> ruleToCostsMap = new HashMap<String, Double>();
		
		for ( Rule rule : availableRules) {
			String ruleName = rule.getClass().getSimpleName().replaceAll("Impl", "");
			ruleToCostsMap.put(ruleName, rule.getCosts());
		}
		
		compToRulesMap.put(affectedComponent, ruleToCostsMap);
		issueToCompToRulesMap.put(issueName, compToRulesMap);
		shopToissueToCompToRulesMap.put(shopName, issueToCompToRulesMap);
		
		try {
			ObjectMapper mapper = new ObjectMapper();
			mapper.writeValue(jsonFile.toFile(), shopToissueToCompToRulesMap);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	

	/**
	 * Selects the component type that is an alternative to the type of the
	 * given affected component and that provides the best reliability.
	 * 
	 * @param affectedComponent
	 *            the given affected component.
	 * @return the selected alternative component type, or <code>null</code> if
	 *         an alternative component type cannot be found.
	 */
	public static ComponentType selectAlternativeComponentType(Component affectedComponent) {
		if (isFilterComponent(affectedComponent)) {
			// a filter component of a certain type cannot be replaced with a
			// filter component of another type. A filter component of the other
			// type can already be deployed, which results in deploying two (or
			// more) components of the same type. Ideally, all filter components
			// types are instantiated once to components and deployed in each
			// shop. When a filter component is removed (crashes), it should be
			// replaced with a component of the same component type. Thus,
			// alternative component types do not apply to filter component
			// types.
			LOGGER.info("There is no alternative component type for the affected component " + affectedComponent);
			return null;
		} else {
			ComponentType affectedComponentType = affectedComponent.getType();
			List<InterfaceType> providedInterfaceTypes = affectedComponentType.getProvidedInterfaceTypes();
			// find all component types that provide (at least) the same
			// interface
			// types as the affected component type.
			Architecture architecture = affectedComponent.getTenant().getArchitecture();
			List<ComponentType> alternativeComponentTypes = new LinkedList<>();
			for (ComponentType ct : architecture.getComponentTypes()) {
				// System.out.print("\n"+"all types "+" :"+ct.getName());
				// System.out.print("\n"+"affected types "+"
				// :"+affectedComponentType.getName());
				// alternative type should be different than the affected type.
				if (!ct.equals(affectedComponentType)) {
					List<InterfaceType> altProvidedInterfaceTypes = ct.getProvidedInterfaceTypes();
					if (altProvidedInterfaceTypes.containsAll(providedInterfaceTypes)) {
						// ct is an alternative component type of the affected
						// component type!
						alternativeComponentTypes.add(ct);
					}
				}
			}

			if (alternativeComponentTypes.size() > 0) {
				// update reliability information
				// ReliabilityManager.updateReliability(alternativeComponentTypes);
				// Use reliability information from the model

				// select component type with the highest reliability
				ComponentType selectedComponentType = alternativeComponentTypes.get(0);
				for (int i = 0; i < alternativeComponentTypes.size(); i++) {
					// System.out.print("\n"+"ALT type "+i+"
					// :"+alternativeComponentTypes.get(i).getName());
					ComponentType alternativeComponentType = alternativeComponentTypes.get(i);
					if (alternativeComponentType.getReliability() > selectedComponentType.getReliability()) {
						selectedComponentType = alternativeComponentType;
					}
				}
				LOGGER.info(selectedComponentType + " has been selected as an alternative for the affected component "
						+ affectedComponent);
				return selectedComponentType;
			} else {
				// no alternative type exists
				LOGGER.info("There is no alternative component type for the affected component " + affectedComponent);
				return null;
			}
		}
	}

	private static boolean isFilterComponent(Component component) {
		ComponentType componentType = component.getType();
		return componentType.getName().contains("Item Filter");
	}

	/*
	 * STATIC AND SEMI_SATIC APPRAOCHES
	 * ===============================================================
	 */

	public static void semiStaticPredictUtility(Issue issue) {
		Rule rule = issue.getHandledBy().get(0);
		double utilityIncrease = 0;
		if (rule instanceof ReplaceComponent) {
			ReplaceComponent replaceComponent = (ReplaceComponent) rule;
			ComponentType alternative = replaceComponent.getAlternativeComponentType();
			double reliability = alternative.getReliability();

			Component affectedComponent = issue.getAffectedComponent();
			// 1-to-1 relationship between a required interface and its
			// connector
			// first, init with number outgoing connectors
			int numberOfConnectors = affectedComponent.getRequiredInterfaces().size();
			// second, add number of incoming connectors
			for (ProvidedInterface pi : affectedComponent.getProvidedInterfaces()) {
				numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
			}

			utilityIncrease = reliability * affectedComponent.getType().getCriticality() * numberOfConnectors;
			// System.out.print("\n"+"reliability"+reliability+"\n");
		} else {
			utilityIncrease = issue.getUtilityDrop();
		}
		rule.setUtilityIncrease(utilityIncrease);
		LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
	}
	///////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * RANDOM APPRAOCH - ALL but RANDOM
	 * ===============================================================
	 */

	public static void R_ALLbutRandomPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}

		for (Rule rule : issue.getHandledBy()) {

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.
			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					utilityIncrease = altComponentType.getReliability() * affectedComponent.getType().getCriticality()
							* affectedComponent.getPerformance()
							* issue.getAffectedComponent().getTenant().getImportance()
							* (affectedComponent.getRequiredInterfaces().size() + 1);

					int numberOfProvided = 0;
					for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
						numberOfProvided = numberOfProvided + pi.getConnectors().size();
					}

					if (numberOfProvided > 1) {
						utilityIncrease = utilityIncrease * numberOfProvided
								- 10 * issue.getAffectedComponent().getADT();
					}

					else if (numberOfProvided == 1) {
						utilityIncrease = utilityIncrease * numberOfProvided * 2
								- 10 * issue.getAffectedComponent().getADT();
					}

					else {
						utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
					} // if(numberOfProvided ==0) utility increase does not
						// change

				}
			} else if (rule instanceof RemoveReplica) {

				double replica = 1;
				double newPerformance = 0;
				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));
				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				if (replica > 0) {
					replica = originalReplica - 1;
					newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
							* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
									* (replica / issue.getAffectedComponent().getRequest()));

					if (0.05 < ((oldPerformance - newPerformance) / oldPerformance))
						while (0.05 < (oldPerformance - newPerformance)) { // System.out.print("\n
																			// stuck
																			// here
																			// 1");
							if (replica > 0) {
								replica--;
								oldPerformance = newPerformance;
								newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
										* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
												* (replica / issue.getAffectedComponent().getRequest()));

							}
						} // end while
				} // end if
				utilityIncrease = issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality() * newPerformance
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);

				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}
				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

			} else if (rule instanceof AddReplica) {

				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				double replica = originalReplica + 1;

				double newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
						* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
								* (replica / issue.getAffectedComponent().getRequest()));

				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));

				// System.out.print("\n stuck here2 ");
				if (0.01 < ((newPerformance - oldPerformance) / newPerformance))
					while (0.01 < ((newPerformance - oldPerformance) / newPerformance)) {// System.out.print("\n
																							// stuck
																							// here
																							// 3");
						replica++;
						// System.out.print("\n stuck here4 ");
						oldPerformance = newPerformance;
						newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
								* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
										* (replica / issue.getAffectedComponent().getRequest()));

					} // end while

				// System.out.print("\n AD Replica " + replica);
				utilityIncrease = issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality() * newPerformance
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);

				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}

				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

			} else {
				utilityIncrease = issue.getUtilityDrop();
			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	/*
	 * RANDOM APPRAOCH - ALL
	 * ===============================================================
	 */

	public static void R_ALLPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}

		double ADT = issue.getAffectedComponent().getADT();
		double[] alphaFactor = new double[3]; // utility increase factors for
		// restart, LW, HW
		Random rn = new Random();
		double chance = rn.nextDouble();
		if (ADT > 1.34) { // Restart has higher chance to be the best, then LW
			// and last HW
			if (chance < 0.55)
				alphaFactor = new double[] { 2.2, 2.1, 2.0 }; // make Restart
			// best with 55
			// percent
			// chance
			else if (chance < 0.85)
				alphaFactor = new double[] { 2.1, 2.2, 2.0 }; // make LW best
			// with 30
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.1, 2.0, 2.2 }; // make HW best
			// with 15
			// percent
			// chance

		} else if (ADT > 1.17) {// LW has higher chance to be the best, then
			// Restart and HW are equal
			if (chance < 0.50)
				alphaFactor = new double[] { 2.1, 2.2, 2.0 }; // make LW best
			// with 50
			// percent
			// chance
			else if (chance < 0.75)
				alphaFactor = new double[] { 2.0, 2.1, 2.2 }; // make HW best
			// with 25
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.2, 2.1, 2.0 }; // make Restart
			// best with 25
			// percent
			// chance

		} else { // 1<ADT<1.17 // HW has higher chance to be the best, then LW
			// and last Restart
			if (chance < 0.55)
				alphaFactor = new double[] { 2.0, 2.1, 2.2 }; // make HW best
			// with 55
			// percent
			// chance
			else if (chance < 0.85)
				alphaFactor = new double[] { 2.0, 2.2, 2.1 }; // make LW best
			// with 30
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.2, 2.0, 2.1 }; // make Restart
			// best with 15
			// percent
			// chance
		}
		// System.out.print("\n alphaFactor "+Arrays.toString(alphaFactor)+"\n\n
		// ___________________________________________________");
		for (Rule rule : issue.getHandledBy()) {

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.
			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectRandomAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					utilityIncrease = 2.1 * altComponentType.getReliability()
							* affectedComponent.getType().getCriticality() * affectedComponent.getPerformance()
							* issue.getAffectedComponent().getTenant().getImportance()
							* (affectedComponent.getRequiredInterfaces().size() + 1);

					int numberOfProvided = 0;
					for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
						numberOfProvided = numberOfProvided + pi.getConnectors().size();
					}

					if (numberOfProvided > 1) {
						utilityIncrease = utilityIncrease * numberOfProvided
								- 10 * issue.getAffectedComponent().getADT();
					}

					else if (numberOfProvided == 1) {
						utilityIncrease = utilityIncrease * numberOfProvided * 2
								- 10 * issue.getAffectedComponent().getADT();
					}

					else {
						utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
					} // if(numberOfProvided ==0) utility increase does not
						// change

				}
			} else if (rule instanceof RestartComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[0];

			} else if (rule instanceof LwRedeployComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[1];

			} else if (rule instanceof HwRedeployComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[2];

			} else if (rule instanceof RemoveReplica) {
				utilityIncrease = 1 * issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality()
						* issue.getAffectedComponent().getType().getPerformanceMax()
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);
				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}
				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

			} else if (rule instanceof AddReplica) {
				utilityIncrease = 1 * issue.getAffectedComponent().getType().getReliability()
						* issue.getAffectedComponent().getType().getCriticality()
						* issue.getAffectedComponent().getType().getPerformanceMax()
						* issue.getAffectedComponent().getTenant().getImportance()
						* (issue.getAffectedComponent().getRequiredInterfaces().size() + 1);

				int numberOfProvided = 0;
				for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
					numberOfProvided = numberOfProvided + pi.getConnectors().size();
				}

				if (numberOfProvided > 1) {
					utilityIncrease = utilityIncrease * numberOfProvided - 10 * issue.getAffectedComponent().getADT();
				}

				else if (numberOfProvided == 1) {
					utilityIncrease = utilityIncrease * numberOfProvided * 2
							- 10 * issue.getAffectedComponent().getADT();
				}

				else {
					utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
				} // if(numberOfProvided ==0) utility increase does not change

			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////

	/*
	 * RANDOM APPRAOCH - SATURATING
	 * ===============================================================
	 */

	public static void R_Nonlinear_SATURPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}
		for (Rule rule : issue.getHandledBy()) {
			// Each rule increases the utility as much as the issue has
			// decreased it, except of the ReplaceComponent rule that may
			// further increase the utility when selecting a component of a
			// "better" alternative type.

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.
			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors

					utilityIncrease = altComponentType.getReliability() * affectedComponent.getType().getCriticality()
							* numberOfConnectors * issue.getAffectedComponent().getPerformance();

				}
			} else if (rule instanceof RemoveReplica) {
				double replica = 1;
				double newPerformance = 0;
				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));
				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				if (replica > 0) {
					replica = originalReplica - 1;
					newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
							* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
									* (replica / issue.getAffectedComponent().getRequest()));

					if (0.05 < ((oldPerformance - newPerformance) / oldPerformance))
						while (0.05 < (oldPerformance - newPerformance)) {
							if (replica > 0) {
								replica--;
								oldPerformance = newPerformance;
								newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
										* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
												* (replica / issue.getAffectedComponent().getRequest()));

							}
						} // end while
				} // end if
					// System.out.print("\n Rm Replica " + replica);

				utilityIncrease = issue.getAffectedComponent().getType().getCriticality()
						* issue.getAffectedComponent().getType().getReliability() * numberOfConnectors * newPerformance;
			} else if (rule instanceof AddReplica) {
				double originalReplica = issue.getAffectedComponent().getInUseReplica();
				double replica = originalReplica + 1;

				double newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
						* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
								* (replica / issue.getAffectedComponent().getRequest()));

				double oldPerformance = issue.getAffectedComponent().getType().getPerformanceMax() * Math.tanh((4
						/ issue.getAffectedComponent().getType().getSatPoint())
						* (issue.getAffectedComponent().getInUseReplica() / issue.getAffectedComponent().getRequest()));

				if (0.01 < ((newPerformance - oldPerformance) / newPerformance))
					while (0.01 < ((newPerformance - oldPerformance) / newPerformance)) {
						replica++;
						oldPerformance = newPerformance;
						newPerformance = issue.getAffectedComponent().getType().getPerformanceMax()
								* Math.tanh((4 / issue.getAffectedComponent().getType().getSatPoint())
										* (replica / issue.getAffectedComponent().getRequest()));

					} // end while

				// System.out.print("\n AD Replica " + replica);

				utilityIncrease = issue.getAffectedComponent().getType().getCriticality()
						* issue.getAffectedComponent().getType().getReliability() * numberOfConnectors * newPerformance;

			} else {
				utilityIncrease = issue.getUtilityDrop();
			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	/*
	 * RANDOM APPRAOCH - DISCONTINOIUS
	 * ===============================================================
	 */

	public static void R_discontinousPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}

		for (Rule rule : issue.getHandledBy()) {
			// Each rule increases the utility as much as the issue has
			// decreased it, except of the ReplaceComponent rule that may
			// further increase the utility when selecting a component of a
			// "better" alternative type.

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.

			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);
					System.out.print("\n  ------\n reliability " + altComponentType.getReliability() + "\n");
					Component affectedComponent = issue.getAffectedComponent();
					utilityIncrease = altComponentType.getReliability() * affectedComponent.getType().getCriticality()
							* issue.getAffectedComponent().getTenant().getImportance()
							* (affectedComponent.getRequiredInterfaces().size() + 1);

					int numberOfProvided = 0;
					for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
						numberOfProvided = numberOfProvided + pi.getConnectors().size();
					}

					if (numberOfProvided > 1) {
						utilityIncrease = utilityIncrease * numberOfProvided
								- 10 * issue.getAffectedComponent().getADT();
					}

					else if (numberOfProvided == 1) {
						utilityIncrease = utilityIncrease * numberOfProvided * 2
								- 10 * issue.getAffectedComponent().getADT();
					}

					else {
						utilityIncrease = utilityIncrease - 10 * issue.getAffectedComponent().getADT();
					}

				}
			} else {
				utilityIncrease = issue.getUtilityDrop();

			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	/*
	 * RANDOM APPRAOCH - LINEAR
	 * ===============================================================
	 */

	public static void R_linearPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}

		for (Rule rule : issue.getHandledBy()) {
			// Each rule increases the utility as much as the issue has
			// decreased it, except of the ReplaceComponent rule that may
			// further increase the utility when selecting a component of a
			// "better" alternative type.

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.
			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors

					utilityIncrease = altComponentType.getReliability() * affectedComponent.getType().getCriticality()
							* numberOfConnectors;

				}
			} else {
				utilityIncrease = issue.getUtilityDrop();
			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	/*
	 * RANDOM APPRAOCH - PROBABILISTIC
	 * ===============================================================
	 */

	public static void R_probabilisticPredictUtility(Issue issue) {
		LOGGER.info("Predicting utility increase for the " + issue.getHandledBy().size()
				+ " rules attached to the issue " + issue);
		// first, add the number of outgoing connectors
		double numberOfConnectors = issue.getAffectedComponent().getRequiredInterfaces().size();
		// second, add number of incoming connectors
		for (ProvidedInterface pi : issue.getAffectedComponent().getProvidedInterfaces()) {
			numberOfConnectors = numberOfConnectors + pi.getConnectors().size();
		}
		double ADT = issue.getAffectedComponent().getADT();
		double[] alphaFactor = new double[3]; // utility increase factors for
		// restart, LW, HW
		Random rn = new Random();
		double chance = rn.nextDouble();
		if (ADT > 1.34) { // Restart has higher chance to be the best, then LW
			// and last HW
			if (chance < 0.55)
				alphaFactor = new double[] { 2.2, 2.1, 2.0 }; // make Restart
			// best with 55
			// percent
			// chance
			else if (chance < 0.85)
				alphaFactor = new double[] { 2.1, 2.2, 2.0 }; // make LW best
			// with 30
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.1, 2.0, 2.2 }; // make HW best
			// with 15
			// percent
			// chance

		} else if (ADT > 1.17) {// LW has higher chance to be the best, then
			// Restart and HW are equal
			if (chance < 0.50)
				alphaFactor = new double[] { 2.1, 2.2, 2.0 }; // make LW best
			// with 50
			// percent
			// chance
			else if (chance < 0.75)
				alphaFactor = new double[] { 2.0, 2.1, 2.2 }; // make HW best
			// with 25
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.2, 2.1, 2.0 }; // make Restart
			// best with 25
			// percent
			// chance

		} else { // 1<ADT<1.17 // HW has higher chance to be the best, then LW
			// and last Restart
			if (chance < 0.55)
				alphaFactor = new double[] { 2.0, 2.1, 2.2 }; // make HW best
			// with 55
			// percent
			// chance
			else if (chance < 0.85)
				alphaFactor = new double[] { 2.0, 2.2, 2.1 }; // make LW best
			// with 30
			// percent
			// chance
			else
				alphaFactor = new double[] { 2.2, 2.0, 2.1 }; // make Restart
			// best with 15
			// percent
			// chance
		}
		// System.out.print("\n alphaFactor "+Arrays.toString(alphaFactor)+"\n\n
		// ___________________________________________________");
		for (Rule rule : issue.getHandledBy()) {
			// Each rule increases the utility as much as the issue has
			// decreased it, except of the ReplaceComponent rule that may
			// further increase the utility when selecting a component of a
			// "better" alternative type.

			double utilityIncrease = 0;
			// to avoid computing the increase for cases that rule Replace
			// component has been selected for other component types then the
			// Authentication Type.
			if (rule instanceof ReplaceComponent
					& issue.getAffectedComponent().getType().getName().equals("Authentication Service")) {
				// select type of the alternative component;
				ComponentType altComponentType = selectRandomAlternativeComponentType(issue.getAffectedComponent());
				if (altComponentType == null) {
					// no alternative component type available. The
					// ReplaceComponent rule cannot be executed. Hence, the
					// utility increase is 0.
					utilityIncrease = 0;

				} else {
					// set the relationship of ReplaceComponent
					ReplaceComponent replaceComponent = (ReplaceComponent) rule;
					replaceComponent.setAlternativeComponentType(altComponentType);

					Component affectedComponent = issue.getAffectedComponent();
					// 1-to-1 relationship between a required interface and its
					// connector
					// first, init with number outgoing connectors

					// System.out.print("\n reliability for replace in
					// Authentication "+
					// altComponentType.getReliability()+"\n"+"___________________________________________________");
					utilityIncrease = 2.1 * altComponentType.getReliability()
							* affectedComponent.getType().getCriticality() * numberOfConnectors
							- 10 * issue.getAffectedComponent().getADT();

				}
			} else if (rule instanceof RestartComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[0];

			} else if (rule instanceof LwRedeployComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[1];

			} else if (rule instanceof HwRedeployComponent) {
				utilityIncrease = issue.getUtilityDrop() * alphaFactor[2];
			}

			rule.setUtilityIncrease(utilityIncrease);
			LOGGER.info("Predicted utility increase of " + utilityIncrease + " for the rule " + rule);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Randomly Selects the component type that is an alternative to the type of
	 * the given affected component
	 * 
	 * @param affectedComponent
	 *            the given affected component.
	 * @return the selected alternative component type, or <code>null</code> if
	 *         an alternative component type cannot be found.
	 */
	public static ComponentType selectRandomAlternativeComponentType(Component affectedComponent) {
		if (isFilterComponent(affectedComponent)) {
			// a filter component of a certain type cannot be replaced with a
			// filter component of another type. A filter component of the other
			// type can already be deployed, which results in deploying two (or
			// more) components of the same type. Ideally, all filter components
			// types are instantiated once to components and deployed in each
			// shop. When a filter component is removed (crashes), it should be
			// replaced with a component of the same component type. Thus,
			// alternative component types do not apply to filter component
			// types.
			LOGGER.info("There is no alternative component type for the affected component " + affectedComponent);
			return null;
		} else {
			ComponentType affectedComponentType = affectedComponent.getType();
			List<InterfaceType> providedInterfaceTypes = affectedComponentType.getProvidedInterfaceTypes();
			// find all component types that provide (at least) the same
			// interface
			// types as the affected component type.
			Architecture architecture = affectedComponent.getTenant().getArchitecture();
			List<ComponentType> alternativeComponentTypes = new LinkedList<>();
			for (ComponentType ct : architecture.getComponentTypes()) {
				// System.out.print("\n"+"all types "+" :"+ct.getName());
				// System.out.print("\n"+"affected types "+"
				// :"+affectedComponentType.getName());
				// alternative type should be different than the affected type.
				if (!ct.equals(affectedComponentType)) {
					List<InterfaceType> altProvidedInterfaceTypes = ct.getProvidedInterfaceTypes();
					if (altProvidedInterfaceTypes.containsAll(providedInterfaceTypes)) {
						// ct is an alternative component type of the affected
						// component type!
						alternativeComponentTypes.add(ct);
					}
				}
			}

			if (alternativeComponentTypes.size() > 0) {
				// update reliability information
				// ReliabilityManager.updateReliability(alternativeComponentTypes);
				// Use reliability information from the model

				// Randomly select component type form index (0) to (size-1)
				int bound = alternativeComponentTypes.size();
				int rnd = new Random().nextInt(bound);
				ComponentType selectedComponentType = alternativeComponentTypes.get(rnd);
				// System.out.print("\n selectedALTComponentType.getReliability
				// "+selectedComponentType.getReliability()+"\n");
				// for (int i = 0; i < alternativeComponentTypes.size(); i++)
				LOGGER.info(selectedComponentType + " has been selected as an alternative for the affected component "
						+ affectedComponent);
				return selectedComponentType;
			} else {
				// no alternative type exists
				LOGGER.info("There is no alternative component type for the affected component " + affectedComponent);
				return null;
			}
		}
	}

}
