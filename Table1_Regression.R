####Libraries and working directories####
library(plyr)
library(lme4)
library(lmerTest)
library(dplyr)
library(MuMIn)
options(scipen=9)
set.seed(42)


setwd("./dataset/")

# AnnotatorWithAttitudes
data=read.csv("./AnnotatorWithAttitudes/largeScale.csv")
#data$toyou <- factor(data$toyou, ordered = TRUE, levels = c("1", "2", "3", "4", "5"))

#data <- data %>%
#  filter(!apply(data, 1, function(x) any(x == "na" | x == "other")))


mod_unsampled <- lmer(toyou  ~ C(annotatorRace)+ C(annotatorPolitics)+ C(annotatorGender) + C(empathy) + C(racism)+ C(harmHateSpeech)+C(lingPurism)+C(freeSpeech)+C(traditionalism) +C(altruism)+(1|tweet) , data = data)
summary(mod_unsampled)
r2 <- r2(mod_unsampled)
print(r2)

dataset=read.csv("./POPQURON/offensiveness/raw_data.csv")

mod_unsampled <- lmer(offensiveness ~ C(race) + C(gender) + C(age) + C(occupation) + C(education) + 
                        (1 | text), data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)


# This is POPQURON politeness

dataset=read.csv("./POPQURON/politeness_rating/raw_data.csv")
#dataset <- dataset %>%
#  filter(!apply(dataset, 1, function(x) any(x == "Prefer not to disclose" | x == "Other")))

mod_unsampled <- lmer(politeness ~ C(race) + C(gender) + C(age) + C(occupation) + C(education) + 
                        (1 | text), data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)


# Now let's check Kumar et al
dataset=read.csv("./Kumar/processed.csv")

mod_unsampled <- lmer(toxic_score ~ C(race) + C(gender)+ C(political_affilation) + C(technology_impact)+ C(is_parent)+C(education) +  C(age_range)   + C(uses_media_social) + C(uses_media_news) +C(religion_important) + C(toxic_comments_problem) +C(uses_media_forums)+C(lgbtq_status) + C(identify_as_transgender) + C(uses_media_video)
                      +(1 | comment), data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)


# Diaz et al. sentiment:
data=read.csv("./DiazSentiment/merged.csv")
mapping <- c("Very negative" = 1, "Somewhat negative" = 2, "Neutral"=3, "Somewhat positive"= 4, "Very positive"=5)
data$annotation <- mapping[data$annotation]
all_vars <- names(data)
formula_vars <- paste0("C(", all_vars, ")")
formula1<- paste(formula_vars, collapse = " + ")
formula1 <- "annotation~ (1|unit_text) + C(Please.indicate.your.age) + C(Please.indicate.your.race...Selected.Choice) + C(Are.you.Hispanic.or.Latino.) + C(How.would.you.describe.the.area.where.you.grew.up.) + C(How.would.you.describe.the.area.in.which.you.currently.live.) + C(In.which.region.of.the.United.States.do.you.currently.live.) + C(Please.indicate.your.annual.household.income) + C(Education) + C(Which.employment.status.best.describes.you.) + C(How.would.you.describe.your.current.living.situation..check.all.that.apply....Selected.Choice)  + C(Please.indicate.your.political.identification) + C(Please.indicate.your.gender) + C(Do.you.consider.yourself.to.be.an.older.adult.) + C(I.have.heard._______________.about.age.discrimination.) + C(In.recent.years.I.have.been.discriminated.against.or.treated.more.negatively.by.others.because.of.my.older.age.) + C(In.recent.years.I.have.avoided.certain.social.settings.out.of.concern.that.I.might.be.treated.negatively.because.of.my.age.) + C(Age.discrimination.is.too.often.excused.) + C(Age.discrimination.is.a.major.problem.in.our.society.) + C(I.have.had.experiences.with.age.discrimination.that.have.caused.me.mental.or.emotional.stress.) + C(Age.discrimination.is.taken.too.seriously.) + C(Have.you.or.anyone.you.know.ever.seen.or.been.the.target.of.discrimination.in.the.workplace.based.on.older.age.) + C(Have.you.or.anyone.you.know.ever.seen.or.been.the.target.of.discrimination.based.on.older.age.while.applying.or.interviewing.for.jobs.) + C(Have.you.or.anyone.you.know.ever.seen.discriminatory.depictions.of.older.age.in.product.marketing.or.advertisements.) + C(Have.you.or.anyone.you.know.ever.seen.discriminatory.depictions.of.older.age.in.television.shows.or.movies.) + C(How.acceptable.do.you.believe.it.is.for.organizations.to.use.age.related.information.in.algorithms.to.show.you.targeted.ads.for.products.and.services.) + C(How.acceptable.do.you.believe.it.is.for.organizations.to.use.age.related.information.in.algorithms.to.review.employment.applications.) + C(How.acceptable.do.you.believe.it.is.for.organizations.to.use.age.related.information.in.algorithms.to.review.applications.for.social.services.) + C(Have.you.ever.been.aware.of.an.automated.algorithm.used.to.review.an.application.of.yours.for.a.job.or.employment.) + C(Have.you.ever.been.aware.of.an.automated.algorithm.used.to.present.you.with.a.product.advertisement.) + C(Have.you.ever.been.aware.of.an.automated.algorithm.used.to.review.an.application.of.yours.for.a.public.or.social.service.) + C(I.enjoy.being.around.old.people.) + C(I.fear.that.when.I.am.old.all.of.my.friends.will.be.gone.) + C(I.like.to.go.visit.my.older.relatives.) + C(I.have.never.lied.about.my.age.in.order.to.appear.younger.) + C(I.fear.it.will.be.very.hard.for.me.to.find.contentment.in.old.age.) + C(The.older.I.become..the.more.I.worry.about.my.health.) + C(I.will.have.plenty.to.occupy.my.time.when.I.am.old.) + C(I.get.nervous.when.I.think.about.someone.else.making.decisions.for.me.when.I.am.old.) + C(It.doesn.t.bother.me.at.all.to.imagine.myself.as.being.old.) + C(I.enjoy.talking.with.old.people.) + C(I.expect.to.feel.good.about.life.when.I.am.old.) + C(I.have.never.dreaded.the.day.I.would.look.in.the.mirror.and.see.gray.hairs.) + C(I.feel.very.comfortable.when.I.am.around.an.old.person.) + C(I.worry.that.people.will.ignore.me.when.I.am.old.) + C(I.have.never.dreaded.looking.old.) + C(I.believe.that.I.will.still.be.able.to.do.most.things.for.myself.when.I.am.old.) + C(I.am.afraid.that.there.will.be.no.meaning.in.life.when.I.am.old.) + C(I.expect.to.feel.good.about.myself.when.I.am.old.) + C(I.enjoy.doing.things.for.old.people.) + C(When.I.look.in.the.mirror..it.bothers.me.to.see.how.my.looks.have.changed.with.age.) + anxiety_score"
mod_unsampled<-lmer(formula1,data=data)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)



#social chemistry 101
dataset=read.csv("./social-chem-101/social-chem-101-processed.csv")
mod_unsampled <- lmer(rot.agree	 ~ C(race) + C(gender)+ C(age)+ C(marital)+C(economic)+C(school)+C(income)+C(children)+(household)+C(us)+C(state)+C(area_y)+C(time.in.us)
                      +(1 | rot), data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)

#NLPositionality

# Try NLPositionality
dataset=read.csv("./NLPositionality/nlpositionality_toxicity_processed.csv")
#dataset <- dataset %>% 
#  filter(!apply(dataset, 1, function(x) any(x == "None")))
mod_unsampled <- lmer(litw	 ~ C(ethnicity) + C(gender)+ C(age)+ C(religion)+C(education)+C(country_longest)+C(country_residence)+C(native_language)+(1 | action),data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)

dataset=read.csv("./NLPositionality/nlpositionality_social-acceptability_processed.csv")
#dataset <- dataset %>% 
#  filter(!apply(dataset, 1, function(x) any(x == "None")))
mod_unsampled <- lmer(litw	 ~ C(ethnicity) + C(gender)+ C(age)+ C(religion)+C(education)+C(country_longest)+C(country_residence)+(1 | action),data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)



# SBIC
dataset=read.csv("./SBIC/SBIC.v2.trn.csv")
mod_unsampled <- lmer(offensiveYN	 ~ C(annotatorRace) + C(annotatorGender)+ C(annotatorAge)+ C(annotatorPolitics)+C(annotatorMinority)+(1 | post),data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)


# EPIC
dataset=read.csv(("./EPIC/EPICorpus.csv"))
dataset$label <- ifelse(dataset$label == "iro", 1, 0)
dataset$label <- as.numeric(dataset$label)
mod_unsampled <- lmer(label	 ~ C(Ethnicity.simplified) + C(Sex)+ (Age)+ C(Country.of.birth)+C(Country.of.residence)+C(Nationality)+C(Student.status)+C(Employment.status)+(1 | text),data = dataset)
print(summary(mod_unsampled))
r2 <- r2(mod_unsampled)
print(r2)


# ANES
# Now let's proceed with ANES
anes2012=read.csv(("./ANES/full_results_2012_2.csv"))
names(anes2012)
#Give intuitive names
anes2012=plyr::rename(anes2012, replace=c("presvote2012_x" = "anes2012_vote", "dem_raceeth_x"="race", 
                                          "discuss_disc"="discuss_politics", "libcpre_self"="ideology",
                                          "pid_x"="party_id", "relig_church"="attend_church",
                                          "dem_age_r_x"="age", "gender_respondent_x"="gender", "paprofile_interestpolit"="interest",
                                          "patriot_flag"="flag", "sample_stfips"="state"))

#Create a binary vote for the anes2012
anes2012$vote_romney[anes2012$anes2012_vote==2]=1
anes2012$vote_romney[anes2012$anes2012_vote==1]=0

#We can also turn GPT-3's predictions into something binary
anes2012$gpt3_vote[anes2012$p_romney>0.5]=1
anes2012$gpt3_vote[anes2012$p_romney<0.5]=0

#Make all negative values on the anes2012 variables missing
anes2012$race[anes2012$race<0]=NA
# I have to remove other race from this 
anes2012$race[anes2012$race>5]=NA

anes2012$discuss_politics[anes2012$discuss_politics<0]=NA
anes2012$ideology[anes2012$ideology<0]=NA
anes2012$party_id[anes2012$party_id<0]=NA
anes2012$attend_church[anes2012$attend_church<0]=NA
anes2012$age[anes2012$age<0]=NA
anes2012$gender[anes2012$gender<0]=NA
anes2012$interest[anes2012$interest<0]=NA
anes2012$flag[anes2012$flag<0]=NA
anes2012$state[anes2012$state<0]=NA

####Let's restrict just to those for whom we have voting data
anes20122=anes2012[!is.na(anes2012$vote_romney),]
anes20122 <- anes20122[complete.cases(anes20122),]

lm1 = lm(vote_romney~C(race)+C(discuss_politics)+C(ideology)+C(party_id)+C(attend_church)+age+C(gender)+C(interest)+C(flag)+C(state), data = anes20122) #Create the linear regression
summary(lm1)


