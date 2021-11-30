package de.mdelab.morisia.selfhealing.rules;

import java.util.Comparator;

import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.Rule;

/**
 * "Note: this comparator imposes orderings that are inconsistent with equals."
 * 
 * @author thomas
 *
 */
public class IssueComparator implements Comparator<Issue> {

	@Override
	public int compare(Issue issue_1, Issue issue_2) {

		// each issue has one rule attached to it
		Rule rule_1 = issue_1.getHandledBy().get(0);
		double value_1 = rule_1.getUtilityIncrease();///rule_1.getCosts();

		Rule rule_2 = issue_2.getHandledBy().get(0);
		double value_2 = rule_2.getUtilityIncrease();///rule_2.getCosts();

		if (value_1 < value_2) {
			return 1;
		} else if (value_1 == value_2) {
			return 0;
		} else {
			// value_1 > value_2;
			return -1;
		}
	}
}
