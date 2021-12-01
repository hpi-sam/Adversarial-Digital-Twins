package de.mdelab.morisia.selfhealing;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.resource.Resource;

import de.mdelab.mlsdm.Activity;
import de.mdelab.mlsdm.interpreter.MLSDMInterpreter;
import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.CF2;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.sdm.interpreter.core.SDMException;
import de.mdelab.expressions.interpreter.core.variables.Variable;

public class IncrementalTest {

	public static void main(String[] args) throws SDMException {

		/*
		 * Init model and SDM interpreter
		 */
		EnvSetUp.initialize();

		/*
		 * Load model
		 */
		Resource architectureResource = EnvSetUp
				.loadInstance("model/mRUBiS-100_enriched.comparch");
		Architecture eArchitecture = (Architecture) architectureResource
				.getContents().get(0);

		Annotations annotations = eArchitecture.getAnnotations();
		if (annotations == null) {
			annotations = ComparchFactory.eINSTANCE.createAnnotations();
			eArchitecture.setAnnotations(annotations);
		}

		/*
		 * load rules
		 */
		Activity analyze_CF2 = EnvSetUp
				.getStoryDiagramActivityFromFile("model/analysis/A_CF2.mlsdm");
		Activity planning_CF2 = EnvSetUp
				.getStoryDiagramActivityFromFile("model/planning/P_CF2.mlsdm");
		Activity execute_cf2_r1 = EnvSetUp
				.getStoryDiagramActivityFromFile("model/Execute/E_CF2.mlsdm");

		/*
		 * get SDM interpreter
		 */
		MLSDMInterpreter interpreter = EnvSetUp
				.getStoryDiagramInterpreter(true);

		// get the params
		List<ProvidedInterface> pInterfaces = new LinkedList<ProvidedInterface>();

		for (Tenant shop : eArchitecture.getTenants()) {
			for (Component c : shop.getComponents()) {
				for (ProvidedInterface pInterface : c.getProvidedInterfaces()) {
					if (pInterface.getFailures().size() >= 5) {
						pInterfaces.add(pInterface);
					}
				}
			}
		}

		// Activity test =
		// getStoryDiagramActivityFromFile("model/Execute/adaptation/RecreateConnector.mlsdm");
		// Component c1 =
		// eArchitecture.getShops().get(0).getComponents().get(10);
		// EcoreUtil.delete(c1.getRequiredInterfaces().get(0).getConnector(),
		// true);
		// System.out.println(c1);
		// Collection<Variable<EClassifier>> ps = new
		// ArrayList<Variable<EClassifier>>();
		//
		// Variable<EClassifier> p = new Variable<EClassifier>("component",
		// c1.eClass(), c1);
		// ps.add(p);
		// interpreter.executeActivity(test, ps);

		/*
		 * ANALYZE
		 */
		for (ProvidedInterface pInterface : pInterfaces) {

			/*
			 * Parameter stuff
			 */
			Collection<Variable<EClassifier>> parameters = new ArrayList<Variable<EClassifier>>();
			Variable<EClassifier> param1 = new Variable<EClassifier>(
					"providedInterface", pInterface.eClass(), pInterface);
			parameters.add(param1);
			Variable<EClassifier> param2 = new Variable<EClassifier>(
					"annotations", annotations.eClass(), annotations);
			parameters.add(param2);

			/*
			 * Execute diagram and get results
			 */
			// Map<String, Variable<EClassifier>> retValues =
			interpreter.executeActivity(analyze_CF2, parameters);

		}

		/*
		 * RULE MATCH
		 */
		for (Issue i : annotations.getIssues()) {
			System.err.println("%%%" + i);
			if (i instanceof CF2) {
				CF2 cf2 = (CF2) i;
				Collection<Variable<EClassifier>> param = new ArrayList<Variable<EClassifier>>();
				param.add(new Variable<EClassifier>("cf2", cf2.eClass(), cf2));
				param.add(new Variable<EClassifier>("annotations", annotations
						.eClass(), annotations));
				interpreter.executeActivity(planning_CF2, param);
			}

		}

		/*
		 * SIDE EFFECT
		 */
		// Extra list since iterating over annotation.getIssues() breaks
		// since
		// the rule removes issues from annotation.getIssues().
		List<Issue> allIssues = new LinkedList<Issue>();
		allIssues.addAll(annotations.getIssues());
		for (Issue issue : allIssues) {
			if (issue instanceof CF2) {
				Rule rule = issue.getHandledBy().get(0);
				Collection<Variable<EClassifier>> param = new ArrayList<Variable<EClassifier>>();
				param.add(new Variable<EClassifier>("rule", rule.eClass(), rule));
				param.add(new Variable<EClassifier>("cf2", issue.eClass(),
						issue));
				interpreter.executeActivity(execute_cf2_r1, param);
			}
		}

		/*
		 * Save stuff
		 */
		EnvSetUp.save(architectureResource, "model/mRUBiS_RESULT.comparch");

	}

}
