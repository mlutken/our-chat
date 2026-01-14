https://englishgrammarzone.com/positive-comparative-and-superlative/


I am reading a dataset like this:


	data_reader = ParquetReader("hf://datasets/HuggingFaceFW/finewiki/data/ptwiki", limit=1)
	for document in data_reader():
		# do something with document
		print(f"type(document):{type(document)}")
		print(document)


How can I convert each document to json inside that loop


Can you create a json file for download with adjectives showing 3 degrees of comparison from the following table:

TABLE START
good	better	best
bad	worse	worst
little	less	least
many	more	most
far	further	furthest
old	older	oldest
near	nearer	nearest
late	later	latest
early	earlier	earliest
high	higher	highest
low	lower	lowest
deep	deeper	deepest
cheap	cheaper	cheapest
clean	cleaner	cleanest
clear	clearer	clearest
clever	more clever	most clever
common	more common	most common
gentle	more gentle	most gentle
narrow	narrower	most narrow
pleasant    more pleasant	most pleasant
polite	more polite	most polite
quiet	more quiet	most quiet
simple	simpler	simplest
stupid	more stupid	most stupid
subtle	more subtle	most subtle
dry	drier	driest
shy	shyer	shyest
sly	slyer	slyest
spry	spryer	spryest
wry	wryer	wryest
gay	gayer	gayest
gray	grayer	grayest
big	bigger	biggest
fat	fatter	fattest
fit	fitter	fittest
flat	flatter	flattest
hot	hotter	hottest
mad	madder	maddest
red	redder	reddest
sad	sadder	saddest
tan	tanner	tannest
thin	thinner	thinnest
wet	wetter	wettest
elder	elder	eldest
outer	outer	outermost
inner	inner	innermost
upper	upper	uppermost
hind	hinder	hindmost
fore	former	foremost
well-known	better-known	best-known
ill-defined	more ill-defined	most ill-defined
far-reaching	more far-reaching	most far-reaching
long-lasting	more long-lasting	most long-lasting
short-lived	more short-lived	most short-lived
nigh	nigher	nighest
worth	more worth	most worth
loath	more loath	most loath
able	abler	ablest
angry	angrier	angriest
busy	busier	busiest
clumsy	clumsier	clumsiest
cozy	cozier	coziest
crazy	crazier	craziest
creepy	creepier	creepiest
crispy	crispier	crispiest
curly	curlier	curliest
dirty	dirtier	dirtiest
easy	easier	easiest
empty	emptier	emptiest
fancy	fancier	fanciest
friendly	friendlier	friendliest
funny	funnier	funniest
gloomy	gloomier	gloomiest
greedy	greedier	greediest
guilty	guiltier	guiltiest
happy	happier	happiest
healthy	healthier	healthiest
heavy	heavier	heaviest
holy	holier	holiest
hungry	hungrier	hungriest
icy	icier	iciest
juicy	juicier	juiciest
lazy	lazier	laziest
likely	likelier	likeliest
lonely	lonelier	loneliest
lovely	lovelier	loveliest
lucky	luckier	luckiest
messy	messier	messiest
mighty	mightier	mightiest
nasty	nastier	nastiest
needy	needier	neediest
noisy	noisier	noisiest
pretty	prettier	prettiest
ready	readier	readiest
risky	riskier	riskiest
scary	scarier	scariest
sleepy	sleepier	sleepiest
smelly	smellier	smelliest
smoky	smokier	smokiest
snowy	snowier	snowiest
sorry	sorrier	sorriest
spicy	spicier	spiciest
sticky	stickier	stickiest
stinky	stinkier	stinkiest
sunny	sunnier	sunniest
tiny	tinier	tiniest
ugly	uglier	ugliest
windy	windier	windiest
witty	wittier	wittiest
woolly	woollier	woolliest
worthy	worthier	worthiest
small	smaller	smallest
large	larger	largest
bright	brighter	brightest
young	younger	youngest
fast	faster	fastest
high	higher	highest
low	lower	lowest
soft	softer	softest
hard	harder	hardest
short	shorter	shortest
clean	cleaner	cleanest
tall	taller	tallest
smart	smarter	smartest
light	lighter	lightest
fresh	fresher	freshest
close	closer	closest
thick	thicker	thickest
thin	thinner	thinnest
rich	richer	richest
poor	poorer	poorest
sweet	sweeter	sweetest
bitter	bitterer	bitterest
strong	stronger	strongest
weak	weaker	weakest
brave	braver	bravest
kind	kinder	kindest
fine	finer	finest
hot	hotter	hottest
cold	colder	coldest
near	nearer	nearest
heavy	heavier	heaviest
cheap	cheaper	cheapest
expensive	more expensive	most expensive
pretty	prettier	prettiest
ugly	uglier	ugliest
happy	happier	happiest
sad	sadder	saddest
old	older	oldest
friendly	friendlier	friendliest
angry	angrier	angriest
bitter	more bitter	most bitter
lazy	lazier	laziest
noisy	noisier	noisiest
gentle	gentler	gentlest
proud	prouder	proudest
wise	wiser	wisest
clear	clearer	clearest
bright	brighter	brightest
simple	simpler	simplest
healthy	healthier	healthiest
large	larger	largest
rare	rarer	rarest
common	more common	most common
important	more important	most important
famous	more famous	most famous
dangerous	more dangerous	most dangerous
beautiful	more beautiful	most beautiful
popular	more popular	most popular
powerful	more powerful	most powerful
gentle	gentler	gentlest
wonderful	more wonderful	most wonderful
dull	duller	dullest
loud	louder	loudest
sharp	sharper	sharpest
happy	happier	happiest
sour	sourer	sourest
spicy	spicier	spiciest
big	bigger	biggest
simple	simpler	simplest
bright	brighter	brightest
bitter	bitterer	bitterest
tasty	tastier	tastiest
wealthy	wealthier	wealthiest
smooth	smoother	smoothest
lazy	lazier	laziest
clever	cleverer	cleverest
heavy	heavier	heaviest
dark	darker	darkest
lucky	luckier	luckiest
lively	livelier	liveliest
happy	happier	happiest
dusty	dustier	dustiest
sharp	sharper	sharpest
thin	thinner	thinnest
sad	sadder	saddest
shy	shyer	shyest
grim	grimmer	grimmest
mad	madder	maddest
rough	rougher	roughest
calm	calmer	calmest
great	greater	greatest
fancy	fancier	fanciest
lovely	lovelier	loveliest
clean	cleaner	cleanest
dim	dimmer	dimmest
ripe	riper	ripest
mighty	mightier	mightiest

TABLE END

following this pattern?

JSON PATTERN START

{
    "base": "dark",
    "classes": ["adj"],
    "adj1": "dark",
    "adj2": "darker",
    "adj3": "darkest"
}
JSON PATTERN END




------------

Can you create a json document with verbs from the following table:

TABLE START
accept	accepted	accepted	accepting	accepts
achieve	achieved	achieved	achieving	achieves
add	added	added	adding	adds
admire	admired	admired	admiring	admires
advise	advised	advised	advising	advises
agree	agreed	agreed	agreeing	agrees
allow	allowed	allowed	allowing	allows
amuse	amused	amused	amusing	amuses
analyze	analyzed	analyzed	analyzing	analyzes
announce	announced	announced	announcing	announces
annoy	annoyed	annoyed	annoying	annoys
answer	answered	answered	answering	answers
apologize	apologized	apologized	apologizing	apologizes
appear	appeared	appeared	appearing	appears
appreciate	appreciated	appreciated	appreciating	appreciates
argue	argued	argued	arguing	argues
arrange	arranged	arranged	arranging	arranges
arrive	arrived	arrived	arriving	arrives
ask	asked	asked	asking	asks
attach	attached	attached	attaching	attaches
attack	attacked	attacked	attacking	attacks
attempt	attempted	attempted	attempting	attempts
attend	attended	attended	attending	attends
attract	attracted	attracted	attracting	attracts
avoid	avoided	avoided	avoiding	avoids
bake	baked	baked	baking	bakes
balance	balanced	balanced	balancing	balances
ban	banned	banned	banning	bans
bang	banged	banged	banging	bangs
bare	bared	bared	baring	bares
bathe	bathed	bathed	bathing	bathes
battle	battled	battled	battling	battles
beam	beamed	beamed	beaming	beams
beg	begged	begged	begging	begs
behave	behaved	behaved	behaving	behaves
belong	belonged	belonged	belonging	belongs
blot	blotted	blotted	blotting	blots
boast	boasted	boasted	boasting	boasts
boil	boiled	boiled	boiling	boils
book	booked	booked	booking	books
bore	bored	bored	boring	bores
borrow	borrowed	borrowed	borrowing	borrows
bounce	bounced	bounced	bouncing	bounces
bow	bowed	bowed	bowing	bows
box	boxed	boxed	boxing	boxes
brake	braked	braked	braking	brakes
branch	branched	branched	branching	branches
breathe	breathed	breathed	breathing	breathes
brush	brushed	brushed	brushing	brushes
bubble	bubbled	bubbled	bubbling	bubbles
bump	bumped	bumped	bumping	bumps
burn	burned/burnt	burned/burnt	burning	burns
bury	buried	buried	burying	buries
buzz	buzzed	buzzed	buzzing	buzzes
calculate	calculated	calculated	calculating	calculates
call	called	called	calling	calls
camp	camped	camped	camping	camps
care	cared	cared	caring	cares
carry	carried	carried	carrying	carries
carve	carved	carved	carving	carves
cause	caused	caused	causing	causes
challenge	challenged	challenged	challenging	challenges
change	changed	changed	changing	changes
charge	charged	charged	charging	charges
chase	chased	chased	chasing	chases
cheat	cheated	cheated	cheating	cheats
check	checked	checked	checking	checks
cheer	cheered	cheered	cheering	cheers
chew	chewed	chewed	chewing	chews
chop	chopped	chopped	chopping	chops
claim	claimed	claimed	claiming	claims
clap	clapped	clapped	clapping	claps
clean	cleaned	cleaned	cleaning	cleans
clear	cleared	cleared	clearing	clears
climb	climbed	climbed	climbing	climbs
close	closed	closed	closing	closes
collect	collected	collected	collecting	collects
color	colored	colored	coloring	colors
comb	combed	combed	combing	combs
come	came	come	coming	comes
comfort	comforted	comforted	comforting	comforts
command	commanded	commanded	commanding	commands
communicate	communicated	communicated	communicating	communicates
compare	compared	compared	comparing	compares
compete	competed	competed	competing	competes
complain	complained	complained	complaining	complains
complete	completed	completed	completing	completes
concentrate	concentrated	concentrated	concentrating	concentrates
concern	concerned	concerned	concerning	concerns
confess	confessed	confessed	confessing	confesses
confuse	confused	confused	confusing	confuses
connect	connected	connected	connecting	connects
consider	considered	considered	considering	considers
consist	consisted	consisted	consisting	consists
contain	contained	contained	containing	contains
continue	continued	continued	continuing	continues
cook	cooked	cooked	cooking	cooks
copy	copied	copied	copying	copies
correct	corrected	corrected	correcting	corrects
cough	coughed	coughed	coughing	coughs
count	counted	counted	counting	counts
cover	covered	covered	covering	covers
crash	crashed	crashed	crashing	crashes
crawl	crawled	crawled	crawling	crawls
create	created	created	creating	creates
cross	crossed	crossed	crossing	crosses
cry	cried	cried	crying	cries
cure	cured	cured	curing	cures
curl	curled	curled	curling	curls
curve	curved	curved	curving	curves
cycle	cycled	cycled	cycling	cycles
dam	dammed	dammed	damming	dams
damage	damaged	damaged	damaging	damages
dance	danced	danced	dancing	dances
dare	dared	dared	daring	dares
deal	dealt	dealt	dealing	deals
decide	decided	decided	deciding	decides
decorate	decorated	decorated	decorating	decorates
delay	delayed	delayed	delaying	delays
delight	delighted	delighted	delighting	delights
deliver	delivered	delivered	delivering	delivers
depend	depended	depended	depending	depends
describe	described	described	describing	describes
desert	deserted	deserted	deserting	deserts
deserve	deserved	deserved	deserving	deserves
destroy	destroyed	destroyed	destroying	destroys
detect	detected	detected	detecting	detects
develop	developed	developed	developing	develops
disagree	disagreed	disagreed	disagreeing	disagrees
discover	discovered	discovered	discovering	discovers
dislike	disliked	disliked	disliking	dislikes
divide	divided	divided	dividing	divides
double	doubled	doubled	doubling	doubles
doubt	doubted	doubted	doubting	doubts
drag	dragged	dragged	dragging	drags
drain	drained	drained	draining	drains
dream	dreamed/dreamt	dreamed/dreamt	dreaming	dreams
dress	dressed	dressed	dressing	dresses
drip	dripped	dripped	dripping	drips
drop	dropped	dropped	dropping	drops
drown	drowned	drowned	drowning	drowns
drum	drummed	drummed	drumming	drums
dry	dried	dried	drying	dries
dust	dusted	dusted	dusting	dusts
earn	earned	earned	earning	earns
educate	educated	educated	educating	educates
employ	employed	employed	employing	employs
empty	emptied	emptied	emptying	empties
encourage	encouraged	encouraged	encouraging	encourages
end	ended	ended	ending	ends
enjoy	enjoyed	enjoyed	enjoying	enjoys
enter	entered	entered	entering	enters
entertain	entertained	entertained	entertaining	entertains
escape	escaped	escaped	escaping	escapes
examine	examined	examined	examining	examines
excite	excited	excited	exciting	excites
excuse	excused	excused	excusing	excuses
exercise	exercised	exercised	exercising	exercises
exist	existed	existed	existing	exists
expand	expanded	expanded	expanding	expands
expect	expected	expected	expecting	expects
explain	explained	explained	explaining	explains
explode	exploded	exploded	exploding	explodes
extend	extended	extended	extending	extends
face	faced	faced	facing	faces
fade	faded	faded	fading	fades
fail	failed	failed	failing	fails
fasten	fastened	fastened	fastening	fastens
fax	faxed	faxed	faxing	faxes
fear	feared	feared	fearing	fears
fence	fenced	fenced	fencing	fences
fetch	fetched	fetched	fetching	fetches
file	filed	filed	filing	files
fill	filled	filled	filling	fills
film	filmed	filmed	filming	films
fire	fired	fired	firing	fires
fit	fitted	fitted	fitting	fits
fix	fixed	fixed	fixing	fixes
flap	flapped	flapped	flapping	flaps
flash	flashed	flashed	flashing	flashes
float	floated	floated	floating	floats
flood	flooded	flooded	flooding	floods
flow	flowed	flowed	flowing	flows
flower	flowered	flowered	flowering	flowers
fold	folded	folded	folding	folds
follow	followed	followed	following	follows
fool	fooled	fooled	fooling	fools
force	forced	forced	forcing	forces
form	formed	formed	forming	forms
found	founded	founded	founding	founds
frame	framed	framed	framing	frames
frighten	frightened	frightened	frightening	frightens
fry	fried	fried	frying	fries
gather	gathered	gathered	gathering	gathers
gaze	gazed	gazed	gazing	gazes
glow	glowed	glowed	glowing	glows
glue	glued	glued	gluing	glues
grab	grabbed	grabbed	grabbing	grabs
grate	grated	grated	grating	grates
grease	greased	greased	greasing	greases
greet	greeted	greeted	greeting	greets
grin	grinned	grinned	grinning	grins
grip	gripped	gripped	gripping	grips
groan	groaned	groaned	groaning	groans
guarantee	guaranteed	guaranteed	guaranteeing	guarantees
guard	guarded	guarded	guarding	guards
guess	guessed	guessed	guessing	guesses
guide	guided	guided	guiding	guides
hammer	hammered	hammered	hammering	hammers
hand	handed	handed	handing	hands
handle	handled	handled	handling	handles
hang	hanged/hung	hanged/hung	hanging	hangs
happen	happened	happened	happening	happens
harm	harmed	harmed	harming	harms
hate	hated	hated	hating	hates
haunt	haunted	haunted	haunting	haunts
heal	healed	healed	healing	heals
heap	heaped	heaped	heaping	heaps
heat	heated	heated	heating	heats
help	helped	helped	helping	helps
hook	hooked	hooked	hooking	hooks
hop	hopped	hopped	hopping	hops
hope	hoped	hoped	hoping	hopes
hover	hovered	hovered	hovering	hovers
hug	hugged	hugged	hugging	hugs
hum	hummed	hummed	humming	hums
hunt	hunted	hunted	hunting	hunts
hurry	hurried	hurried	hurrying	hurries
identify	identified	identified	identifying	identifies
ignore	ignored	ignored	ignoring	ignores
imagine	imagined	imagined	imagining	imagines
impress	impressed	impressed	impressing	impresses
improve	improved	improved	improving	improves
include	included	included	including	includes
increase	increased	increased	increasing	increases
inform	informed	informed	informing	informs
inject	injected	injected	injecting	injects
injure	injured	injured	injuring	injures
instruct	instructed	instructed	instructing	instructs
intend	intended	intended	intending	intends
interest	interested	interested	interesting	interests
interrupt	interrupted	interrupted	interrupting	interrupts
introduce	introduced	introduced	introducing	introduces
invent	invented	invented	inventing	invents
invite	invited	invited	inviting	invites
irritate	irritated	irritated	irritating	irritates
itch	itched	itched	itching	itches
jail	jailed	jailed	jailing	jails
join	joined	joined	joining	joins
joke	joked	joked	joking	jokes
judge	judged	judged	judging	judges
juggle	juggled	juggled	juggling	juggles
jump	jumped	jumped	jumping	jumps
justify	justified	justified	justifying	justifies
keep	kept	kept	keeping	keeps
kick	kicked	kicked	kicking	kicks
kill	killed	killed	killing	kills
kiss	kissed	kissed	kissing	kisses
kneel	kneeled/knelt	kneeled/knelt	kneeling	kneels
knit	knitted	knitted	knitting	knits
knock	knocked	knocked	knocking	knocks
knot	knotted	knotted	knotting	knots
label	labeled/labelled	labeled/labelled	labeling/labelling	labels/labels
land	landed	landed	landing	lands
last	lasted	lasted	lasting	lasts
laugh	laughed	laughed	laughing	laughs
launch	launched	launched	launching	launches
learn	learned/learnt	learned/learnt	learning	learns
level	leveled/levelled	leveled/levelled	leveling/levelling	levels/levels
license	licensed/licenced	licensed/licenced	licensing/licencing	licenses/licences
lick	licked	licked	licking	licks
lie	lied	lied	lying	lies
light	lighted/lit	lighted/lit	lighting	lights
like	liked	liked	liking	likes
list	listed	listed	listing	lists
listen	listened	listened	listening	listens
live	lived	lived	living	lives
load	loaded	loaded	loading	loads
locate	located	located	locating	locates
lock	locked	locked	locking	locks
long	longed	longed	longing	longs
look	looked	looked	looking	looks
love	loved	loved	loving	loves
maintain	maintained	maintained	maintaining	maintains
manage	managed	managed	managing	manages
march	marched	marched	marching	marches
mark	marked	marked	marking	marks
marry	married	married	marrying	marries
match	matched	matched	matching	matches
mate	mated	mated	mating	mates
matter	mattered	mattered	mattering	matters
measure	measured	measured	measuring	measures
meddle	meddled	meddled	meddling	meddles
melt	melted	melted	melting	melts
memorize	memorized	memorized	memorizing	memorizes
mend	mended	mended	mending	mends
mention	mentioned	mentioned	mentioning	mentions
mess up	messed up	messed up	messing up	messes up
milk	milked	milked	milking	milks
mine	mined	mined	mining	mines
miss	missed	missed	missing	misses
mix	mixed	mixed	mixing	mixes
moan	moaned	moaned	moaning	moans
mourn	mourned	mourned	mourning	mourns
move	moved	moved	moving	moves
multiply	multiplied	multiplied	multiplying	multiplies
murder	murdered	murdered	murdering	murders
nail	nailed	nailed	nailing	nails
name	named	named	naming	names
need	needed	needed	needing	needs
nest	nested	nested	nesting	nests
nod	nodded	nodded	nodding	nods
note	noted	noted	noting	notes
notice	noticed	noticed	noticing	notices
number	numbered	numbered	numbering	numbers
obey	obeyed	obeyed	obeying	obeys
object	objected	objected	objecting	objects
observe	observed	observed	observing	observes
obtain	obtained	obtained	obtaining	obtains
occur	occurred	occurred	occurring	occurs
offend	offended	offended	offending	offends
offer	offered	offered	offering	offers
open	opened	opened	opening	opens
order	ordered	ordered	ordering	orders
overflow	overflowed	overflowed	overflowing	overflows
owe	owed	owed	owing	owes
own	owned	owned	owning	owns
pack	packed	packed	packing	packs
paddle	paddled	paddled	paddling	paddles
paint	painted	painted	painting	paints
park	parked	parked	parking	parks
part	parted	parted	parting	parts
pass	passed	passed	passing	passes
paste	pasted	pasted	pasting	pastes
pat	patted	patted	patting	pats
pause	paused	paused	pausing	pauses
pedal	pedaled/pedalled	pedaled/pedalled	pedaling/pedalling	pedals/pedals
peep	peeped	peeped	peeping	peeps
perform	performed	performed	performing	performs
permit	permitted	permitted	permitting	permits
phone	phoned	phoned	phoning	phones
pick	picked	picked	picking	picks
pinch	pinched	pinched	pinching	pinches
pine	pined	pined	pining	pines
place	placed	placed	placing	places
plan	planned	planned	planning	plans
plant	planted	planted	planting	plants
play	played	played	playing	plays
please	pleased	pleased	pleasing	pleases
plug	plugged	plugged	plugging	plugs
point	pointed	pointed	pointing	points
poke	poked	poked	poking	pokes
polish	polished	polished	polishing	polishes
pop	popped	popped	popping	pops
possess	possessed	possessed	possessing	possesses
post	posted	posted	posting	posts
pour	poured	poured	pouring	pours
practise	practised	practised	practising	practises
praise	praised	praised	praising	praises
pray	prayed	prayed	praying	prays
preach	preached	preached	preaching	preaches
precede	preceded	preceded	preceding	precedes
prefer	preferred	preferred	preferring	prefers
prepare	prepared	prepared	preparing	prepares
present	presented	presented	presenting	presents
preserve	preserved	preserved	preserving	preserves
press	pressed	pressed	pressing	presses
pretend	pretended	pretended	pretending	pretends
prevent	prevented	prevented	preventing	prevents
prick	pricked	pricked	pricking	pricks
print	printed	printed	printing	prints
produce	produced	produced	producing	produces
program	programmed	programmed	programming	programs
promise	promised	promised	promising	promises
protect	protected	protected	protecting	protects
qualify	qualified	qualified	qualifying	qualifies
question	questioned	questioned	questioning	questions
quit	quit/quitted	quit/quitted	quitting	quits/quits
rain	rained	rained	raining	rains
raise	raised	raised	raising	raises
reach	reached	reached	reaching	reaches
realize	realized	realized	realizing	realizes
receive	received	received	receiving	receives
recognize	recognized	recognized	recognizing	recognizes
recommend	recommended	recommended	recommending	recommends
record	recorded	recorded	recording	records
reduce	reduced	reduced	reducing	reduces
reflect	reflected	reflected	reflecting	reflects
refuse	refused	refused	refusing	refuses
regard	regarded	regarded	regarding	regards
relate	related	related	relating	relates
relax	relaxed	relaxed	relaxing	relaxes
rely	relied	relied	relying	relies
remain	remained	remained	remaining	remains
remember	remembered	remembered	remembering	remembers
remind	reminded	reminded	reminding	reminds
remove	removed	removed	removing	removes
repair	repaired	repaired	repairing	repairs
repeat	repeated	repeated	repeating	repeats
replace	replaced	replaced	replacing	replaces
reply	replied	replied	replying	replies
report	reported	reported	reporting	reports
represent	represented	represented	representing	represents
request	requested	requested	requesting	requests
rescue	rescued	rescued	rescuing	rescues
research	researched	researched	researching	researches
reserve	reserved	reserved	reserving	reserves
reside	resided	resided	residing	resides
resolve	resolved	resolved	resolving	resolves
respond	responded	responded	responding	responds
restore	restored	restored	restoring	restores
retire	retired	retired	retiring	retires
return	returned	returned	returning	returns
reveal	revealed	revealed	revealing	reveals
reward	rewarded	rewarded	rewarding	rewards
ride	rode	ridden	riding	rides
ring	rang	rung	ringing	rings
rise	rose	risen	rising	rises
risk	risked	risked	risking	risks
roar	roared	roared	roaring	roars
rob	robbed	robbed	robbing	robs
roll	rolled	rolled	rolling	rolls
rotate	rotated	rotated	rotating	rotates
rub	rubbed	rubbed	rubbing	rubs
ruin	ruined	ruined	ruining	ruins
rule	ruled	ruled	ruling	rules
run	ran	run	running	runs
rush	rushed	rushed	rushing	rushes
satisfy	satisfied	satisfied	satisfying	satisfies
save	saved	saved	saving	saves
saw	sawed	sawn/sawed	sawing	saws
say	said	said	saying	says
scare	scared	scared	scaring	scares
scatter	scattered	scattered	scattering	scatters
scold	scolded	scolded	scolding	scolds
scream	screamed	screamed	screaming	screams
search	searched	searched	searching	searches
see	saw	seen	seeing	sees
seek	sought	sought	seeking	seeks
select	selected	selected	selecting	selects
sell	sold	sold	selling	sells
send	sent	sent	sending	sends
separate	separated	separated	separating	separates
serve	served	served	serving	serves
set	set	set	setting	sets
settle	settled	settled	settling	settles
sew	sewed	sewn/sewed	sewing	sews
shake	shook	shaken	shaking	shakes
shape	shaped	shaped	shaping	shapes
share	shared	shared	sharing	shares
shave	shaved	shaved	shaving	shaves
sheathe	sheathed	sheathed	sheathing	sheathes
shelter	sheltered	sheltered	sheltering	shelters
shine	shined	shined/shone	shining	shines
ship	shipped	shipped	shipping	ships
shock	shocked	shocked	shocking	shocks
shoot	shot	shot	shooting	shoots
shout	shouted	shouted	shouting	shouts
show	showed	shown	showing	shows
shrink	shrank	shrunk	shrinking	shrinks
shut	shut	shut	shutting	shuts
sigh	sighed	sighed	sighing	sighs
sign	signed	signed	signing	signs
signal	signaled/signalled	signaled/signalled	signaling/signalling	signals/signals
simplify	simplified	simplified	simplifying	simplifies
sing	sang	sung	singing	sings
sink	sank	sunk	sinking	sinks
sit	sat	sat	sitting	sits
sketch	sketched	sketched	sketching	sketches
ski	skied	skied	skiing	skis
skip	skipped	skipped	skipping	skips
slap	slapped	slapped	slapping	slaps
sleep	slept	slept	sleeping	sleeps
slice	sliced	sliced	slicing	slices
slide	slid	slid	sliding	slides
slip	slipped	slipped	slipping	slips
smile	smiled	smiled	smiling	smiles
smoke	smoked	smoked	smoking	smokes
snap	snapped	snapped	snapping	snaps
sniff	sniffed	sniffed	sniffing	sniffs
snore	snored	snored	snoring	snores
snow	snowed	snowed	snowing	snows
soak	soaked	soaked	soaking	soaks
solve	solved	solved	solving	solves
soothe	soothed	soothed	soothing	soothes
sort	sorted	sorted	sorting	sorts
sound	sounded	sounded	sounding	sounds
sow	sowed	sown/sowed	sowing	sows
spare	spared	spared	sparing	spares
spark	sparked	sparked	sparking	sparks
speak	spoke	spoken	speaking	speaks
spell	spelled/spelt	spelled/spelt	spelling	spells
spend	spent	spent	spending	spends
spill	spilled/spilt	spilled/spilt	spilling	spills
spin	spun	spun	spinning	spins
spit	spat	spat	spitting	spits
talk	talked	talked	talking	talks
tame	tamed	tamed	taming	tames
tap	tapped	tapped	tapping	taps
target	targeted	targeted	targeting	targets
taste	tasted	tasted	tasting	tastes
tease	teased	teased	teasing	teases
tell	told	told	telling	tells
tempt	tempted	tempted	tempting	tempts
terrify	terrified	terrified	terrifying	terrifies
thank	thanked	thanked	thanking	thanks
thaw	thawed	thawed	thawing	thaws
tick	ticked	ticked	ticking	ticks
tickle	tickled	tickled	tickling	tickles
tie	tied	tied	tying	ties
time	timed	timed	timing	times
tip	tipped	tipped	tipping	tips
tire	tired	tired	tiring	tires
touch	touched	touched	touching	touches
tour	toured	toured	touring	tours
tow	towed	towed	towing	tows
trace	traced	traced	tracing	traces
track	tracked	tracked	tracking	tracks
trade	traded	traded	trading	trades
train	trained	trained	training	trains
transport	transported	transported	transporting	transports
trap	trapped	trapped	trapping	traps
travel	traveled/travelled	traveled/travelled	traveling/travelling	travels/travels
treat	treated	treated	treating	treats
tremble	trembled	trembled	trembling	trembles
trick	tricked	tricked	tricking	tricks
trip	tripped	tripped	tripping	trips
trot	trotted	trotted	trotting	trots
trouble	troubled	troubled	troubling	troubles
trust	trusted	trusted	trusting	trusts
try	tried	tried	trying	tries
tug	tugged	tugged	tugging	tugs
tumble	tumbled	tumbled	tumbling	tumbles
turn	turned	turned	turning	turns
twist	twisted	twisted	twisting	twists
type	typed	typed	typing	types
uncover	uncovered	uncovered	uncovering	uncovers
undress	undressed	undressed	undressing	undresses
unfasten	unfastened	unfastened	unfastening	unfastens
unite	united	united	uniting	unites
unlock	unlocked	unlocked	unlocking	unlocks
unpack	unpacked	unpacked	unpacking	unpacks
untidy	untidied	untidied	untidying	untidies
use	used	used	using	uses
vanish	vanished	vanished	vanishing	vanishes
visit	visited	visited	visiting	visits
vex	vexed	vexed	vexing	vexes
wait	waited	waited	waiting	waits
walk	walked	walked	walking	walks
wander	wandered	wandered	wandering	wanders
want	wanted	wanted	wanting	wants
warm	warmed	warmed	warming	warms
warn	warned	warned	warning	warns
wash	washed	washed	washing	washes
waste	wasted	wasted	wasting	wastes
watch	watched	watched	watching	watches
water	watered	watered	watering	waters
wave	waved	waved	waving	waves
weigh	weighed	weighed	weighing	weighs
welcome	welcomed	welcomed	welcoming	welcomes
whine	whined	whined	whining	whines
whip	whipped	whipped	whipping	whips
whirl	whirled	whirled	whirling	whirls
whisper	whispered	whispered	whispering	whispers
whistle	whistled	whistled	whistling	whistles
wink	winked	winked	winking	winks
wipe	wiped	wiped	wiping	wipes
wish	wished	wished	wishing	wishes
wobble	wobbled	wobbled	wobbling	wobbles
wonder	wondered	wondered	wondering	wonders
work	worked	worked	working	works
worry	worried	worried	worrying	worries
wrap	wrapped	wrapped	wrapping	wraps
wreck	wrecked	wrecked	wrecking	wrecks
wrestle	wrestled	wrestled	wrestling	wrestles
wriggle	wriggled	wriggled	wriggling	wriggles
x-ray	x-rayed	x-rayed	x-raying	x-rays
yawn	yawned	yawned	yawning	yawns
yell	yelled	yelled	yelling	yells
yelp	yelped	yelped	yelping	yelps
yield	yielded	yielded	yielding	yields
zip	zipped	zipped	zipping	zips
zoom	zoomed	zoomed	zooming	zooms

TABLE END

following this pattern?


JSON PATTERN START

{
    "base": "accept",
    "classes": ["v"],
    "v1": "accept",
    "v2": "accepted",
    "v3": "accepted",
    "v4": "accepting",
    "v5": "accepts"
}

JSON PATTERN END



