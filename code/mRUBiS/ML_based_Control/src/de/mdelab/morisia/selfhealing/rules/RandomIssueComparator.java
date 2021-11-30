package de.mdelab.morisia.selfhealing.rules;

import java.util.Comparator;
import java.util.Random;


import de.mdelab.morisia.comparch.Issue;

public class RandomIssueComparator implements Comparator<Issue> {

	
	@Override
	public int compare(Issue issue_1, Issue issue_2) {
		

int max=1;
int min =-1;

    Random  rn=new Random();
    int result = rn.nextInt((max-min)+1)+min;
    return result;
		
	}
}
