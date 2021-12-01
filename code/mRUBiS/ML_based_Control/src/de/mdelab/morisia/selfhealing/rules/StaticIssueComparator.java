package de.mdelab.morisia.selfhealing.rules;

import java.util.Comparator;

import de.mdelab.morisia.comparch.Issue;

public class StaticIssueComparator implements Comparator<Issue> {

	@Override
	public int compare(Issue issue_1, Issue issue_2) {
		
		String issue_1_name = issue_1.eClass().getName();
		String issue_2_name = issue_2.eClass().getName();


		if (issue_1_name.equals("CF3")) {
			return -1;
		} else if (issue_2_name.equals("CF3")) {
			return 1;
		} else if (issue_1_name.equals("CF1") & issue_2_name.equals("CF2")) {
			return -1;
		}  else if (issue_2_name.equals("CF1") & issue_1_name.equals("CF2")) {
			return 1;
		} else {
			return 0;
		}
	}
}
