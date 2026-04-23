# v10_tokens soft-preprocess regression audit (2026-03-12)

Compared runs:
- Baseline: `eval_results/adapters_v10_tokens_v7_quick/2026-03-12_120341_test_labeled_student_v10_tokens_0000600.*`
- Soft preprocess: `eval_results/adapters_v10_tokens_v7_quick/2026-03-12_141418_test_labeled.preprocessed_soft_student_v10_tokens_0000600.*`

Headline:
- Shared valid rows: 236
- Regressions (baseline correct -> soft wrong): 27
- Improvements (baseline wrong -> soft correct): 15
- Net on shared rows: -12 correct
- Baseline had 3 invalid rows; soft had 0. Soft got 2/3 of those correct, giving +2 back.
- Net final effect: -10 correct overall (49.6% -> 44.8% label accuracy).

Regression pattern counts (27 jobs):
- `tech_shift`: 19
- `tech_expanded`: 16
- `loc_shift`: 10
- `comp_shift`: 13
- `comp_no_gbp_lost`: 11
- `arr_shift`: 3
- `sen_shift`: 1
- `heavy_truncation`: 16
- `removed_keyworded_lines`: 22

## Job-by-job regressions
### #15 Technical Director
- job_id: `4355949740`
- label: `bad_fit` | baseline `bad_fit` (0) -> soft `maybe` (60)
- jd length: 6845 -> 3192 (drop 3653)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, loc_shift, arr_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=UNK | base=OUTSIDE_UK -> soft=REMOTE
  - arr: gold=REMOTE | base=HYBRID -> soft=REMOTE
  - tech: gold=['JS_TS', 'NODE', 'REACT'] | base=['JS_TS', 'REACT'] -> soft=['JS_TS', 'NODE']
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - Are you a hands-on leader who specializes in solution and application architecture?

### #159 Creator Partnerships & Business Development Executive
- job_id: `https://uk.linkedin.com/jobs/view/creator-partnerships-business-development-executive-at-everpress-4375682601`
- label: `bad_fit` | baseline `bad_fit` (0) -> soft `maybe` (60)
- jd length: 5115 -> 3200 (drop 1915)
- tags: loc_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=UK_OTHER | base=OUTSIDE_UK -> soft=UK_OTHER
- removed keyword-bearing snippets:
  - We believefashion can—and should—be kinder to the planet and even kinder to people.Everpress is a global platform where independent artists, creators, brands, andcommunities lau...

### #38 Graduate Software Developer
- job_id: `4375810033`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `good_fit` (75)
- jd length: 4390 -> 2784 (drop 1606)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, loc_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=IN_LONDON | base=UK_OTHER -> soft=IN_LONDON
  - tech: gold=['JS_TS'] | base=['JS_TS', 'NODE'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=RANGE_55_74K | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - RequirementsWhat we're looking forA degree in Computer Science, Software Engineering, or a related field (Bachelor's, Master's or PhD)Recent graduates or 1-2 years working in in...
  - Benefits🎓 Ready to build your future?If you're excited by impactful work, fast learning, and the chance to help shape the future of frontline work — we'd love to hear from you.B...

### #236 Software Engineer - New Grad
- job_id: `https://www.linkedin.com/jobs/view/software-engineer-new-grad-at-scale-ai-4297642437`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `good_fit` (75)
- jd length: 5372 -> 3180 (drop 2192)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, loc_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=OUTSIDE_UK | base=UK_OTHER -> soft=IN_LONDON
  - tech: gold=['AI_ML', 'JS_TS', 'REACT'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - This allows us to ensure a fair and thorough evaluation of all applicants.About Us:At Scale, our mission is to develop reliable AI systems for the world's most important decisions.
  - We work closely with industry leaders like Meta, Cisco, DLA Piper, Mayo Clinic, Time Inc., the Government of Qatar, and U.S.
  - We are expanding our team to accelerate the development of AI applications.We believe that everyone should be able to bring their whole selves to work, which is why we are proud...
  - PLEASE NOTE: We collect, retain and use personal data for our professional business purposes, including notifying you of job opportunities that may be of interest and sharing wi...

### #3 Software Engineer - Backend
- job_id: `4214697219`
- label: `bad_fit` | baseline `bad_fit` (30) -> soft `maybe` (60)
- jd length: 3764 -> 3109 (drop 655)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded
- token changes:
  - tech: gold=['OOS'] | base=['JS_TS'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K

### #183 Junior Developer
- job_id: `https://uk.linkedin.com/jobs/view/junior-developer-at-viajero-rent-a-car-4379518972`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `good_fit` (75)
- jd length: 3296 -> 3188 (drop 108)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, loc_shift, removed_keyworded_lines
- token changes:
  - loc: gold=IN_LONDON | base=UK_OTHER -> soft=IN_LONDON
  - tech: gold=['OOS'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - Highly skilled, passionate, collaborative, and customer-centric we pride ourselves on our people.LNKD1_UKTJ

### #165 Frontend Developer (12-month Fixed-term Contract)
- job_id: `https://uk.linkedin.com/jobs/view/frontend-developer-12-month-fixed-term-contract-at-science-in-sport-group-4357516313`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (60)
- jd length: 6118 -> 2202 (drop 3916)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['JS_TS'] | base=['JS_TS', 'NODE'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=RANGE_45_54K | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - As well as being the Official Energy Gel Provider of RunThrough and the Official Nutrition Partner of Ultra X.The RoleContract: 12-month Fixed-term contractSalary: £40,000 - £50...
  - Alongside this, you will be supported with a personal performance plan that will enable you to continue improving your own performance and impact.Experience and mindset that wil...

### #232 Full-Stack Software Engineer
- job_id: `https://www.linkedin.com/jobs/view/full-stack-software-engineer-at-tulip-interfaces-4254109273`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (60)
- jd length: 5133 -> 3120 (drop 2013)
- tags: tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['AI_ML', 'JS_TS', 'NODE', 'REACT'] | base=['JS_TS', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - Tulip's cloud-native, no-code platform, powered by embedded AI, is driving the digital transformation of industrial environments through composable, human-centric solutions that...

### #161 Data Analyst
- job_id: `https://uk.linkedin.com/jobs/view/data-analyst-at-price-forbes-4322082243`
- label: `maybe` | baseline `maybe` (55) -> soft `good_fit` (75)
- jd length: 5198 -> 3196 (drop 2002)
- tags: comp_shift, comp_no_gbp_lost, loc_shift, heavy_truncation
- token changes:
  - loc: gold=IN_LONDON | base=UK_OTHER -> soft=IN_LONDON
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K

### #178 Graduate Frontend Engineer
- job_id: `https://uk.linkedin.com/jobs/view/graduate-frontend-engineer-at-targetjobs-uk-4374941095`
- label: `bad_fit` | baseline `bad_fit` (35) -> soft `maybe` (55)
- jd length: 3681 -> 3173 (drop 508)
- tags: tech_shift, tech_expanded
- token changes:
  - tech: gold=['JS_TS', 'REACT'] | base=['JS_TS', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']

### #201 Senior Full Stack Software Developer (uk)
- job_id: `https://uk.linkedin.com/jobs/view/senior-full-stack-software-developer-uk-at-keyzo-4380924233`
- label: `maybe` | baseline `maybe` (65) -> soft `good_fit` (80)
- jd length: 5996 -> 1915 (drop 4081)
- tags: tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['JS_TS', 'NODE', 'REACT'] | base=['JS_TS', 'NODE'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - We offer unique product features and first-class service that set us apart in the marketplace.This role is suited to an established senior developer who has spent several years ...

### #185 Junior Full Stack PHP Software Engineer
- job_id: `https://uk.linkedin.com/jobs/view/junior-full-stack-php-software-engineer-at-action-sustainability-4378761310`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (60)
- jd length: 5043 -> 2364 (drop 2679)
- tags: loc_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=IN_LONDON | base=UK_OTHER -> soft=IN_LONDON
- removed keyword-bearing snippets:
  - This comprehensive approach supports an environment where personal and professional growth thrive.About the Role:You’ll work closely with senior engineers to build and improve f...
  - Please refer to their Data Privacy Policy & Notice on their website for further details.

### #61 Senior DevOps Engineer
- job_id: `4380977388_sal_0`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (60)
- jd length: 5773 -> 3146 (drop 2627)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['OOS'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=RANGE_75_99K | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - Job DescriptionGlobal Application Deployment Platform (Windows Based)AeroCloud: Revolutionizing Airport OperationsAeroCloud is the new-age operating system for airports aiming t...
  - Our suite includes Airport Operating Systems (AOS), Passenger Processing Systems (PPS), and Passenger Flow Management solutions, empowering airports to gain deep insights into t...
  - Whether in times of need or growth, we stand alongside our clients, offering support through innovative software that drives their success.We build software that runs live airpo...
  - The focus is applicationdelivery, versioning, and operational correctness at scale.The kind of experience that fits this roleWe’re optimising for people who have done this for r...

### #224 Software Engineer Intern
- job_id: `https://uk.linkedin.com/jobs/view/software-engineer-intern-at-deliveroo-4377291603`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (60)
- jd length: 5459 -> 3121 (drop 2338)
- tags: loc_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - loc: gold=IN_LONDON | base=UK_OTHER -> soft=IN_LONDON
- removed keyword-bearing snippets:
  - Check out our Tech Blog.We aim to create a fair process that lets your skills shine—our interview typically includes 3-4 stages.
  - Depending on your role, you may collaborate with teammates, systems, and leaders across DoorDash and Wolt.
  - We’ll share details on how to request support so we can ensure you have a fair and equitable experience.If you’re excited about making a real impact in a fast-moving marketplace...

### #147 Software Engineer (Entry-Level)
- job_id: `https://ca.linkedin.com/jobs/view/software-engineer-entry-level-at-nutanix-4379201981`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (55)
- jd length: 4434 -> 3194 (drop 1240)
- tags: comp_shift, loc_shift, arr_shift, removed_keyworded_lines
- token changes:
  - loc: gold=OUTSIDE_UK | base=UK_OTHER -> soft=IN_LONDON
  - arr: gold=IN_OFFICE | base=HYBRID -> soft=UNK
  - comp: gold=NO_GBP | base=RANGE_45_54K -> soft=NO_GBP
- removed keyword-bearing snippets:
  - Joining our Software Engineering team at Nutanix will empower you to grow through hands-on mentorship while contributing to systems that power hybrid multicloud environments use...
  - Teams establishing a footprint include those focused on:Kubernetes-based enterprise platformsNetworking and security solutionsDistributed systemsQuality engineering and automati...
  - Being physically present allows for seamless teamwork and direct access to resources that support your success.
  - In good faith, the posting may be removed prior to this date if the position is filled or extended in good faith.

### #235 Software Engineer I
- job_id: `https://www.linkedin.com/jobs/view/software-engineer-i-at-microsoft-4378787058`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (55)
- jd length: 4186 -> 2976 (drop 1210)
- tags: loc_shift, arr_shift, removed_keyworded_lines
- token changes:
  - loc: gold=OUTSIDE_UK | base=UK_OTHER -> soft=IN_LONDON
  - arr: gold=UNK | base=HYBRID -> soft=UNK
- removed keyword-bearing snippets:
  - Each day we build on our values of respect, integrity, and accountability to create a culture of inclusion where everyone can thrive at work and beyond.This role is targeting an...

### #53 Senior Electrical Design Engineer
- job_id: `4378956335_sal_0`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (55)
- jd length: 3832 -> 2934 (drop 898)
- tags: tech_shift, tech_expanded, removed_keyworded_lines
- token changes:
  - tech: gold=['OOS'] | base=['JS_TS', 'NODE'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - People work in a flat structure with hands-on application, innovation and experimentation constantly in mind.Benefits include a competitive basic salary, pension, life insurance...

### #45 Software Engineer
- job_id: `4377786888`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (60)
- jd length: 3940 -> 3175 (drop 765)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, tech_expanded, removed_keyworded_lines
- token changes:
  - tech: gold=['OOS'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
  - comp: gold=RANGE_55_74K | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - We partner with governments to work together towards a common goal, defending our freedom.We are proud of our employee-led networks, examples include: Gender Equality, Pride, Me...

### #204 Senior JavaScript Developer Vue 3
- job_id: `https://uk.linkedin.com/jobs/view/senior-javascript-developer-vue-3-at-client-server-4378749517`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (60)
- jd length: 2767 -> 2767 (drop 0)
- tags: tech_shift, tech_expanded
- token changes:
  - tech: gold=['JS_TS'] | base=['JS_TS', 'NODE'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']

### #55 Senior Engineering Manager
- job_id: `4379245433`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (55)
- jd length: 6363 -> 3193 (drop 3170)
- tags: tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['JS_TS', 'NODE'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - The Engineering Manager will be responsible for leading a team of software engineers in the development and maintenance of Mistplay's main product, our Android app.
  - The successful candidate will have a deep understanding of developing mobile apps/games, as well as experience leading a team of developers in an agile development environment.
  - You will be responsible for providing technical guidance to the team and resolving technical issues as they arise• Partner closely with stakeholders (e.g Product, Data, Executiv...
  - We foster an environment where everyone is encouraged to share their ideas, push boundaries, take calculated risks, and witness their visions come to life.We may use artificial ...

### #148 Senior Software Engineer - Web (m/f/d)
- job_id: `https://de.linkedin.com/jobs/view/senior-software-engineer-web-m-f-d-at-flink-4368019739`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (55)
- jd length: 5585 -> 3142 (drop 2443)
- tags: tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['AI_ML', 'JS_TS', 'NODE', 'REACT'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['AI_ML', 'JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - We carry 2,500 products with a focus on the leading brands in each category.
  - We aim for sustainability by making deliveries on electric bikes and utilizing recyclable packaging.Flink launched in February 2021 and already operates in more than 90 cities i...
  - You will build the features that customers interact with daily, from secure authentication flows to engaging gamification experiences.Lead Complex Feature Delivery: Own the end-...
  - You will tackle challenging problems like building fraud-resistant systems, creating seamless onboarding experiences, and developing self-service profile management.Champion Web...

### #228 Backend Developer
- job_id: `https://www.linkedin.com/jobs/view/backend-developer-at-hudu-4375439728`
- label: `bad_fit` | baseline `bad_fit` (40) -> soft `maybe` (50)
- jd length: 4792 -> 3150 (drop 1642)
- tags: tech_shift, tech_expanded, heavy_truncation, removed_keyworded_lines
- token changes:
  - tech: gold=['JS_TS', 'REACT'] | base=['JS_TS', 'REACT'] -> soft=['JS_TS', 'NODE', 'REACT']
- removed keyword-bearing snippets:
  - Building tools that solve real pain points for users.
  - Benefits & Perks:Health Insurance401k plan with company matchingPaid time-offFlexible work hoursWork Life BalanceCompensation for this role will be determined on several factors...
  - Reasonable accommodations are available upon request throughout the hiring process.At Hudu, we believe in fairness and transparency throughout our hiring process.

### #85 Junior Software Engineer
- job_id: `gen_v7_0089`
- label: `maybe` | baseline `maybe` (60) -> soft `good_fit` (70)
- jd length: 862 -> 859 (drop 3)
- tags: comp_shift, removed_keyworded_lines
- token changes:
  - comp: gold=RANGE_75_99K | base=RANGE_55_74K -> soft=RANGE_75_99K
- removed keyword-bearing snippets:
  - NextGen Solutions is a tech company that specializes in developing AI-driven software solutions for various sectors. Our team is passionate about innovation and is dedicated to ...
  - - Develop and maintain software applications using Node.js and javascript.
  - - Experience with Node.js and JavaScript/TypeScript.
  - - Salary range of £75,000-£95,000 per year.

### #19 Human Resources Manager
- job_id: `4363310160`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (50)
- jd length: 4963 -> 3200 (drop 1763)
- tags: comp_shift, comp_no_gbp_lost, heavy_truncation, removed_keyworded_lines
- token changes:
  - comp: gold=NO_GBP | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - This position is responsible for attracting and retaining top talent in a highly competitive labor market, negotiating and administering medical and benefits plans, and ensuring...

### #43 Data Engineer
- job_id: `4377463363`
- label: `bad_fit` | baseline `bad_fit` (45) -> soft `maybe` (50)
- jd length: 3730 -> 3118 (drop 612)
- tags: tech_shift, tech_expanded
- token changes:
  - tech: gold=['NODE'] | base=['JS_TS', 'NODE'] -> soft=['JS_TS', 'NODE', 'REACT']

### #39 Senior / Lead Software Engineer
- job_id: `4376018297_sal_0`
- label: `good_fit` | baseline `good_fit` (70) -> soft `maybe` (60)
- jd length: 7353 -> 3175 (drop 4178)
- tags: comp_shift, comp_no_gbp_lost, tech_shift, sen_shift, heavy_truncation, removed_keyworded_lines
- token changes:
  - sen: gold=LEVEL_3 | base=LEVEL_3 -> soft=LEVEL_2
  - tech: gold=['AI_ML', 'JS_TS', 'NODE'] | base=['JS_TS', 'NODE', 'REACT'] -> soft=['JS_TS', 'NODE']
  - comp: gold=RANGE_55_74K | base=NO_GBP -> soft=RANGE_55_74K
- removed keyword-bearing snippets:
  - Implement and maintain CI/CD pipelines to streamline the development and deployment process.
  - Contribute to the developer community Inquisitive about internal areas like bids and hiring Provide technical leadership, coaching, and mentoring to your team Promote knowledge ...
  - CI/CD and Automation: Implementing and maintaining continuous integration/continuous deployment pipelines to accelerate development cycles.
  - Accountability: Be accountable for delivering your part of a project on time and under budget and working well with other leaders.

### #101 Security Engineer
- job_id: `gen_v7_0157`
- label: `good_fit` | baseline `good_fit` (70) -> soft `maybe` (60)
- jd length: 1084 -> 1080 (drop 4)
- tags: tech_shift, removed_keyworded_lines
- token changes:
  - tech: gold=['AI_ML', 'JS_TS', 'NODE'] | base=['AI_ML', 'JS_TS', 'NODE'] -> soft=['JS_TS', 'NODE']
- removed keyword-bearing snippets:
  - SecureNet Solutions is a leader in cybersecurity, providing innovative solutions to protect businesses from digital threats. We are looking for a Security Engineer to join our f...
  - - Design and implement secure software solutions using node.js and JS/Typescript.
  - - Proficiency in Node.js, JavaScript, and TypeScript.
  - - Competitive salary of £80k-£100k.
