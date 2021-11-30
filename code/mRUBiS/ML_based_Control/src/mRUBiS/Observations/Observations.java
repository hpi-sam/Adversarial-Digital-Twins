package mRUBiS.Observations;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.stream.Collectors;
import java.util.ArrayList;
import java.util.HashMap;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Capability;
import de.mdelab.morisia.comparch.simulator.ComparchSimulator;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.impl.Trace_1;
import de.mdelab.morisia.selfhealing.ArchitectureUtilCal;
import de.mdelab.morisia.selfhealing.incremental.EventListener;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class Observations {
	
	public static String getNumberOfShops(Architecture MRUBIS) {
		String json = "";
		
		HashMap<String, Integer> numberOfShopsMap = new HashMap<String, Integer>();
		
		numberOfShopsMap.put("number_of_shops", MRUBIS.getTenants().size());
		
		try {
			json = new ObjectMapper().writeValueAsString(numberOfShopsMap);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

		return json;
	}

	public static String getInitialState(Tenant shop) {
		String json = "";

		HashMap<String, HashMap<String, HashMap<String, String>>> shopMap = new HashMap<String, HashMap<String, HashMap<String, String>>>();

		HashMap<String, HashMap<String, String>> componentMap = new HashMap<String, HashMap<String, String>>();
		shopMap.put(shop.getName(), componentMap);

		for (Component component: shop.getComponents()) {
			
			HashMap<String, String> parameterMap = new HashMap<String, String>();
			
			parameterMap.put("uid", component.getUid());
			parameterMap.put("adt", String.valueOf(component.getADT()));
			parameterMap.put("connectivity", String.valueOf(new Double(component.getProvidedInterfaces().size() + component.getRequiredInterfaces().size())));
			parameterMap.put("required_interface", String.valueOf(component.getRequiredInterfaces().size()));
			parameterMap.put("provided_interface", String.valueOf(component.getProvidedInterfaces().size()));
			parameterMap.put("importance", String.valueOf(component.getTenant().getImportance()));
			parameterMap.put("reliability", String.valueOf(component.getType().getReliability()));
			parameterMap.put("criticality", String.valueOf(component.getType().getCriticality()));
			parameterMap.put("request", String.valueOf(component.getRequest()));
			parameterMap.put("sat_point", String.valueOf(component.getType().getSatPoint()));
			parameterMap.put("replica", String.valueOf(component.getInUseReplica()));
			parameterMap.put("perf_max", String.valueOf(component.getType().getPerformanceMax()));
			parameterMap.put("component_utility", String.valueOf(ArchitectureUtilCal.computeComponentUtility(component)));
			parameterMap.put("system_utility", String.valueOf(ArchitectureUtilCal.computeOverallUtility(shop.getArchitecture())));
			parameterMap.put("shop_utility", String.valueOf(ArchitectureUtilCal.computeShopUtility(shop)));
			componentMap.put(component.getType().getName() , parameterMap);
		}

		try {
			json = new ObjectMapper().writeValueAsString(shopMap);
			return json;
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

		return json;

	}
	
	
	public static String getNumberOfIssuesPerShop(Architecture MRUBIS) {
		String json = "";
		
		HashMap<String, Integer> numberOfIssuesMap = new HashMap<String, Integer>();
		
		Integer numberOfIssues = MRUBIS.getAnnotations().getIssues().size();
		numberOfIssuesMap.put("number_of_issues_in_run", numberOfIssues);
		
		try {
			json = new ObjectMapper().writeValueAsString(numberOfIssuesMap);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

		return json;
		
	}
	
	

	public static String getAffectedComponentStatus(Architecture MRUBIS, HashMap<String, HashMap<String, HashMap<String, HashMap<String, Double>>>> issuesToRulesMap){

		String json = "";

		HashMap<String, HashMap<String, HashMap<String, String>>> shopMap = new HashMap<String, HashMap<String, HashMap<String, String>>>();

		for (Tenant shop : MRUBIS.getTenants())
		{
			
			String shopName = shop.getName();
			
			if (issuesToRulesMap.containsKey(shopName)) {
				
				HashMap<String, HashMap<String, String>> componentMap = new HashMap<String, HashMap<String, String>>();
				HashMap<String, HashMap<String, HashMap<String, Double>>> shopIssueToRulesMap = issuesToRulesMap.get(shopName);

				List<Issue> issues = MRUBIS.getAnnotations().getIssues();
				
				System.out.println("Current issuesmap: " + issuesToRulesMap);

				for ( Issue issue: issues) {

					Component affectedComponent = issue.getAffectedComponent();
					String affectedComponentType = affectedComponent.getType().getName();
					String failureName = issue.getClass().getSimpleName().replaceAll("Impl", "");
					
					//System.out.println("Current failure name: " + failureName);
					//System.out.println("Current issueToRuleMap: " + issuesToRulesMap);
					Boolean failureNameInIssuesToRulesMap = shopIssueToRulesMap.containsKey(failureName);
					
					if (failureNameInIssuesToRulesMap && shopIssueToRulesMap.get(failureName).containsKey(affectedComponentType)) {
						
						HashMap<String, Double> availableRules = shopIssueToRulesMap.get(failureName).get(affectedComponentType);
						List<String> availableRuleNames = new ArrayList<>(availableRules.keySet());
						List<String> availableRuleCosts = availableRules.values().stream().map( cost -> String.valueOf(cost) ).collect( Collectors.toList() );

						HashMap<String, String> parameterMap = new HashMap<String, String>();

						parameterMap.put("uid", affectedComponent.getUid());
						parameterMap.put("failure_name", failureName.toString());
						parameterMap.put("rule_names", availableRuleNames.toString());
						parameterMap.put("rule_costs", availableRuleCosts.toString());
						parameterMap.put("adt", String.valueOf(affectedComponent.getADT()));
						parameterMap.put("connectivity", String.valueOf(new Double(affectedComponent.getProvidedInterfaces().size() + affectedComponent.getRequiredInterfaces().size())));
						parameterMap.put("required_interface", String.valueOf(affectedComponent.getRequiredInterfaces().size()));
						parameterMap.put("provided_interface", String.valueOf(affectedComponent.getProvidedInterfaces().size()));
						parameterMap.put("importance", String.valueOf(affectedComponent.getTenant().getImportance()));
						parameterMap.put("reliability", String.valueOf(affectedComponent.getType().getReliability()));
						parameterMap.put("criticality", String.valueOf(affectedComponent.getType().getCriticality()));
						parameterMap.put("request", String.valueOf(affectedComponent.getRequest()));
						parameterMap.put("sat_point", String.valueOf(affectedComponent.getType().getSatPoint()));
						parameterMap.put("replica", String.valueOf(affectedComponent.getInUseReplica()));
						parameterMap.put("perf_max", String.valueOf(affectedComponent.getType().getPerformanceMax()));
						parameterMap.put("component_utility", String.valueOf(ArchitectureUtilCal.computeComponentUtility(affectedComponent)));
						parameterMap.put("system_utility", String.valueOf(ArchitectureUtilCal.computeOverallUtility(shop.getArchitecture())));
						parameterMap.put("shop_utility", String.valueOf(ArchitectureUtilCal.computeShopUtility(shop)));

						componentMap.put(affectedComponentType , parameterMap);
						
					} else {
//						System.out.println("Failed to find rule for issue " + failureName + " affecting component " + affectedComponentType);
//						System.out.println("Current issue to rules map: " + issuesToRulesMap.get(failureName));
//						System.out.println("Current affected component: " + affectedComponentType);
					}
				

				}

				shopMap.put(shop.getName(), componentMap);
				
			}

		}

		try {
			json = new ObjectMapper().writeValueAsString(shopMap);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

		return json;

	}
	
	public static String getFixedComponentStatus(Architecture MRUBIS, HashMap<String, List<String>> fixedComponents){
		
		System.out.println("Components fixed in this run: ");
		for (String shopName: fixedComponents.keySet()) {
			System.out.println(shopName + ": " + fixedComponents.get(shopName));
		};
		System.out.println();

		String json = "";

		HashMap<String, HashMap<String, HashMap<String, String>>> shopMap = new HashMap<String, HashMap<String, HashMap<String, String>>>();

		for (Tenant shop : MRUBIS.getTenants())
		{

			List<Component> shopComponents = shop.getComponents();
			String shopName = shop.getName();
			HashMap<String, HashMap<String, String>> componentMap = new HashMap<String, HashMap<String, String>>();
			
			if (fixedComponents.containsKey(shopName)) {
				
				List<String> fixedComponentsInThisShop = fixedComponents.get(shopName);
				
				for (Component component: shopComponents) {
					
					String componentType =  component.getType().getName();
					HashMap<String, String> parameterMap = new HashMap<String, String>();
					
					if (fixedComponentsInThisShop.contains(componentType)) {
						
						System.out.println(shopName + ": Updating fixed component " + componentType);
						
						// check if the component still has any remaining issues
						String failureName = "";
						String availableRuleNames = "";
						String availableRuleCosts = "";
						if (component.getIssues().size() > 0) {
							System.out.println("Obs: Issue remaining on " + componentType + ": " + failureName);
							Issue persistentIssue = component.getIssues().get(0);
							failureName = persistentIssue.getClass().getSimpleName().replaceAll("Impl", "");
							List<Rule> availableRules = persistentIssue.getHandledBy();
							availableRuleCosts = availableRules.stream().map(rule -> rule.getClass().getSimpleName().replaceAll("Impl", "")).collect( Collectors.toList() ).toString();
							availableRuleCosts = availableRules.stream().map(rule -> String.valueOf(rule.getCosts())).collect( Collectors.toList() ).toString();
						}
						
						parameterMap.put("uid", component.getUid());
						parameterMap.put("failure_name", failureName.toString());
						parameterMap.put("rule_names", availableRuleNames.toString());
						parameterMap.put("rule_costs", availableRuleCosts.toString());
						parameterMap.put("adt", String.valueOf(component.getADT()));
						parameterMap.put("connectivity", String.valueOf(new Double(component.getProvidedInterfaces().size() + component.getRequiredInterfaces().size())));
						parameterMap.put("required_interface", String.valueOf(component.getRequiredInterfaces().size()));
						parameterMap.put("provided_interface", String.valueOf(component.getProvidedInterfaces().size()));
						parameterMap.put("importance", String.valueOf(component.getTenant().getImportance()));
						parameterMap.put("reliability", String.valueOf(component.getType().getReliability()));
						parameterMap.put("criticality", String.valueOf(component.getType().getCriticality()));
						parameterMap.put("request", String.valueOf(component.getRequest()));
						parameterMap.put("sat_point", String.valueOf(component.getType().getSatPoint()));
						parameterMap.put("replica", String.valueOf(component.getInUseReplica()));
						parameterMap.put("perf_max", String.valueOf(component.getType().getPerformanceMax()));
						parameterMap.put("component_utility", String.valueOf(ArchitectureUtilCal.computeComponentUtility(component)));
						parameterMap.put("system_utility", String.valueOf(ArchitectureUtilCal.computeOverallUtility(shop.getArchitecture())));
						parameterMap.put("shop_utility", String.valueOf(ArchitectureUtilCal.computeShopUtility(shop)));
						
						componentMap.put(componentType, parameterMap);
						
					}
					
				}
				
			}
			
			shopMap.put(shop.getName(), componentMap);
			
		}
		
		try {
			json = new ObjectMapper().writeValueAsString(shopMap);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}

		return json;
	}



	public static void makeObservation(Architecture mRUBiS){


		ArchitectureUtilCal.computeOverallUtility(mRUBiS);

		for (Tenant shop : mRUBiS.getTenants())
		{ArchitectureUtilCal.computeShopUtility(shop);

		shop.getName();
		shop.getCriticality();
		shop.getPerformance();
		shop.getRequest();


		for ( Component component : shop.getComponents())
		{    ArchitectureUtilCal.computeComponentUtility(component);
		component.getInUseReplica();
		component.getIssues();
		component.getType();
		component.getCriticality();
		}

		}


		List<Issue> allIssues = new LinkedList<>();
		allIssues.addAll(mRUBiS.getAnnotations().getIssues());
		for (Issue issue : allIssues)
		{issue.getAffectedComponent();
		issue.getUtilityDrop();
		issue.getHandledBy();
		issue.getHandledBy().get(0).getCosts();}

	}
}




