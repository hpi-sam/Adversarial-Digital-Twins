package mRUBiS_Tasks;
import java.util.Comparator;
import de.mdelab.morisia.comparch.Issue;

public class RLIssueComparator implements Comparator<Issue> {

		@Override
		public int compare(Issue issue_1, Issue issue_2) {
			   issue_1.getAffectedComponent();
			    issue_1.getAffectedComponent().getType();
			    issue_1.getAffectedComponent().getTenant().getName();
			
			    issue_2.getAffectedComponent();
			    issue_2.getAffectedComponent().getType();
			    issue_2.getAffectedComponent().getTenant().getName();
			    
			String issue_1_name = issue_1.eClass().getName();
			String issue_2_name = issue_2.eClass().getName();


			if (issue_1_name.equals("CF3")) {
				return -1;
			} else if (issue_2_name.equals("CF3")&&issue_1.getAffectedComponent().getTenant().getName().equals("BidAndBuyService")) {
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





