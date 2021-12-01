package mRUBiS_Tasks;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.Rule;

public class Input {
public static void selectAction(Issue issue) {
		
		String issueName = issue.getClass().getSimpleName().replaceAll("Impl", "");
	    String shopName = issue.getAffectedComponent().getTenant().getName();
	    String componentType = issue.getAffectedComponent().getType().getName();
		
		// Read rules to execute from JSON (received and stored RuleSelector)
		Path rulesToExecutePath= Paths.get("rulesToExecute.json");
		HashMap<String, HashMap<String, HashMap<String, String>>> rulesToExecute = new HashMap<String, HashMap<String, HashMap<String, String>>>();
		try {
			rulesToExecute = new ObjectMapper().readValue(rulesToExecutePath.toFile(), HashMap.class);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String actionToExecute = "";
		actionToExecute = rulesToExecute.get(shopName).get(issueName).get(componentType);
		
		//Remove all the other possible actions
		List<Rule> actionsToRemove= new LinkedList<Rule>();
		List<Rule> actionToKeep = new LinkedList<Rule>();
		for (Rule rule : issue.getHandledBy()) {
			if(rule.eClass().getName().equals(actionToExecute))
				{actionToKeep.add(rule);
				System.out.print("\n\n  Action successfully added to Execution List ");
				}
			else actionsToRemove.add(rule);
				
		}
		
		
		for (Rule rule : actionsToRemove) {
			rule.setAnnotations(null);
			rule.setHandles(null);
		}

	
		
	}

}
