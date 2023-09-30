    public void
    execute(final org.bukkit.command.CommandSender sender, final java.lang.String[] args) {if ((((((!sender.hasPermission("areashop.createrent")) && (!sender.hasPermission("areashop.createrent.member"))) &&  (!sender.hasPermission("areashop.createrent.owner"))) &&  (!sender.hasPermission("areashop.createbuy"))) && (!sender.hasPermission("areashop.createbuy.member")))
    && (!sender.hasPermission("areashop.createbuy.owner"))) { plugin.message(sender, "add-noPermission");

        return;}if (((args.length < 2) || (args[1]
    == null))  || ((!"rent".equalsIgnoreCase(args[1])) && (!"buy".equalsIgnoreCase(args[1])))) {

        plugin.message(sender,
        "add-help");
        return;
        }

        java.util.Map<java.lang.String,
        com.sk89q.worldguard.protection.regions.ProtectedRegion> regions  =  new  java.util.HashMap<>(); org.bukkit.World world;org.bukkit.entity.Player
        player = null;
        if  (sender
        instanceof
        org.bukkit.entity.Player) {
            player = ((org.bukkit.entity.Player) (sender));

    }

    if  (args.length == 2)
    {
    if (player
    == null) {
        plugin.message(sender, "cmd-weOnlyByPlayer");return;
    }

    me.wiefferink.areashop.interfaces.WorldEditSelection selection = plugin.getWorldEditHandler().getPlayerSelection(player);
    if (selection
    == null)
    { plugin.message(player, "cmd-noSelection"); return; }
world
=
selection.getWorld();
regions  = me.wiefferink.areashop.tools.Utils.getWorldEditRegionsInSelection(selection).stream().collect(java.util.stream.Collectors.toMap(ProtectedRegion::getId, (
region) -> region));

if (regions.isEmpty()) {
plugin.message(player, "cmd-noWERegionsFound");return; } } else {if (player
!= null) {
if (args.length ==  4) {
world  = org.bukkit.Bukkit.getWorld(args[3]); if (world  == null)
{ plugin.message(sender,  "add-incorrectWorld",
args[3]);

return;
}
} else
{world
= ((org.bukkit.entity.Player) (sender)).getWorld();
}
}  else
if (args.length < 4) {
plugin.message(sender,  "add-specifyWorld");
return;}  else {

world = org.bukkit.Bukkit.getWorld(args[3]);
if  (world
==  null) {
plugin.message(sender,
"add-incorrectWorld", args[3]);return;
}}
com.sk89q.worldguard.protection.regions.ProtectedRegion region = plugin.getRegionManager(world).getRegion(args[2]);
if (region == null) {
plugin.message(sender,
"cmd-noRegion",
args[2]);return;
}
regions.put(args[2], region);

} final boolean isRent = "rent".equalsIgnoreCase(args[1]); final org.bukkit.entity.Player  finalPlayer = player;me.wiefferink.areashop.AreaShop.debug(("Starting add task with "
+ regions.size())
+
" regions");
java.util.TreeSet<me.wiefferink.areashop.regions.GeneralRegion>
regionsSuccess
= new java.util.TreeSet<>(); java.util.TreeSet<me.wiefferink.areashop.regions.GeneralRegion> regionsAlready = new java.util.TreeSet<>(); java.util.TreeSet<me.wiefferink.areashop.regions.GeneralRegion> v9 = new  java.util.TreeSet<>();
java.util.TreeSet<me.wiefferink.areashop.regions.GeneralRegion> regionsRentCancelled = new
java.util.TreeSet<>();// Denied by an event listener
java.util.TreeSet<me.wiefferink.areashop.regions.GeneralRegion>  v11 =  new
java.util.TreeSet<>();


java.util.TreeSet<java.lang.String>  v12
=  new
java.util.TreeSet<>(); java.util.TreeSet<java.lang.String>  namesNoPermission
= new  java.util.TreeSet<>();
java.util.TreeSet<java.lang.String> namesAddCancelled
= new java.util.TreeSet<>();// Denied by an event listener


me.wiefferink.bukkitdo.Do.forAll(plugin.getConfig().getInt("adding.regionsPerTick"), regions.entrySet(), ( regionEntry) -> {
java.lang.String  regionName =
regionEntry.getKey(); com.sk89q.worldguard.protection.regions.ProtectedRegion region = regionEntry.getValue(); // Determine if the player is an owner or member of the region
boolean isMember =  (finalPlayer  !=  null)
&&  me.wiefferink.areashop.commands.plugin.getWorldGuardHandler().containsMember(region, finalPlayer.getUniqueId());

boolean
isOwner  = (finalPlayer  !=
null)
&& me.wiefferink.areashop.commands.plugin.getWorldGuardHandler().containsOwner(region, finalPlayer.getUniqueId());
java.lang.String v19;if (isRent) {
v19 = "rent";
} else {
v19  = "buy"; }me.wiefferink.areashop.managers.FileManager.AddResult result =  me.wiefferink.areashop.commands.plugin.getFileManager().checkRegionAdd(sender,  region, world,
isRent  ? GeneralRegion.RegionType.RENT : GeneralRegion.RegionType.BUY);if  (result
== FileManager.AddResult.ALREADYADDED)
{
regionsAlready.add(me.wiefferink.areashop.commands.plugin.getFileManager().getRegion(regionName)); } else
if  (result ==
FileManager.AddResult.ALREADYADDEDOTHERWORLD) {

v9.add(me.wiefferink.areashop.commands.plugin.getFileManager().getRegion(regionName)); } else if
(result == FileManager.AddResult.BLACKLISTED)
{ v12.add(regionName);
}
else
if (result
== FileManager.AddResult.NOPERMISSION) {
namesNoPermission.add(regionName); } else {
// Check if the player should be landlord
boolean
v21
= (!sender.hasPermission("areashop.create"  + v19)) && ((sender.hasPermission(("areashop.create" + v19)
+ ".owner") && isOwner) ||
(sender.hasPermission(("areashop.create"
+  v19)
+
".member") &&
isMember));
List<java.util.UUID> existing = new me.wiefferink.areashop.commands.ArrayList<>();

existing.addAll(me.wiefferink.areashop.commands.plugin.getWorldGuardHandler().getOwners(region).asUniqueIdList());
existing.addAll(me.wiefferink.areashop.commands.plugin.getWorldGuardHandler().getMembers(region).asUniqueIdList());
me.wiefferink.areashop.AreaShop.debug("regionAddLandlordStatus:", regionName,
"landlord:", v21,  "existing:",
existing, "isMember:", isMember,
"isOwner:",
isOwner, "createPermission:", sender.hasPermission("areashop.create"
+
v19), "ownerPermission:",  sender.hasPermission(("areashop.create" +  v19)  +
".owner"),  "memberPermission:", sender.hasPermission(("areashop.create" + v19) +  ".member"));

if (isRent) {
me.wiefferink.areashop.regions.RentRegion
rent  =
new me.wiefferink.areashop.regions.RentRegion(regionName,
world);// Set landlord

if  (v21) {

rent.setLandlord(finalPlayer.getUniqueId(), finalPlayer.getName());} me.wiefferink.areashop.events.ask.AddingRegionEvent event = me.wiefferink.areashop.commands.plugin.getFileManager().addRegion(rent);
if (event.isCancelled())
{
namesAddCancelled.add(rent.getName());return; }
rent.handleSchematicEvent(GeneralRegion.RegionEvent.CREATED);
rent.update();
// Add existing owners/members if any
if ((!v21) && (!existing.isEmpty())) {
java.util.UUID rentBy = existing.remove(0);

org.bukkit.OfflinePlayer  v26
= org.bukkit.Bukkit.getOfflinePlayer(rentBy);
me.wiefferink.areashop.events.ask.RentingRegionEvent rentingRegionEvent = new me.wiefferink.areashop.events.ask.RentingRegionEvent(rent, v26,  false);
org.bukkit.Bukkit.getPluginManager().callEvent(rentingRegionEvent);
if  (rentingRegionEvent.isCancelled()) {regionsRentCancelled.add(rent);}
else
{// Add values to the rent and send it to FileManager
rent.setRentedUntil(java.util.Calendar.getInstance().getTimeInMillis()  +  rent.getDuration());
rent.setRenter(rentBy);rent.updateLastActiveTime();

rent.handleSchematicEvent(GeneralRegion.RegionEvent.RENTED);
// Add others as friends
for (java.util.UUID friend
: existing)
{

rent.getFriendsFeature().addFriend(friend,
null);
}

rent.notifyAndUpdate(new me.wiefferink.areashop.events.notify.RentedRegionEvent(rent, false));
}

}
regionsSuccess.add(rent); } else {me.wiefferink.areashop.regions.BuyRegion buy = new me.wiefferink.areashop.regions.BuyRegion(regionName,
world); // Set landlord
if (v21)  {
buy.setLandlord(finalPlayer.getUniqueId(),  finalPlayer.getName());} me.wiefferink.areashop.events.ask.AddingRegionEvent event  =
me.wiefferink.areashop.commands.plugin.getFileManager().addRegion(buy);
if  (event.isCancelled()) {
namesAddCancelled.add(buy.getName());
return;}

buy.handleSchematicEvent(GeneralRegion.RegionEvent.CREATED);
buy.update();
if
((!v21)
&&  (!existing.isEmpty())) {
java.util.UUID  buyBy = existing.remove(0);
org.bukkit.OfflinePlayer
buyByPlayer = org.bukkit.Bukkit.getOfflinePlayer(buyBy);
me.wiefferink.areashop.events.ask.BuyingRegionEvent buyingRegionEvent  =  new me.wiefferink.areashop.events.ask.BuyingRegionEvent(buy, buyByPlayer);
org.bukkit.Bukkit.getPluginManager().callEvent(buyingRegionEvent);
if (buyingRegionEvent.isCancelled()) {
v11.add(buy);
} else {

buy.setBuyer(buyBy);
buy.updateLastActiveTime();


buy.handleSchematicEvent(GeneralRegion.RegionEvent.BOUGHT);

for (java.util.UUID
v34 : existing)
{

buy.getFriendsFeature().addFriend(v34, null);}
buy.notifyAndUpdate(new me.wiefferink.areashop.events.notify.BoughtRegionEvent(buy));
}
}

regionsSuccess.add(buy);
}}
},
() -> { if  (!regionsSuccess.isEmpty()) {me.wiefferink.areashop.commands.plugin.message(sender,
"add-success", args[1],  me.wiefferink.areashop.tools.Utils.combinedMessage(regionsSuccess,
"region")); }if
(!regionsAlready.isEmpty()) {me.wiefferink.areashop.commands.plugin.message(sender, "add-failed",  me.wiefferink.areashop.tools.Utils.combinedMessage(regionsAlready, "region"));
}
if (!v9.isEmpty()) {
me.wiefferink.areashop.commands.plugin.message(sender, "add-failedOtherWorld", me.wiefferink.areashop.tools.Utils.combinedMessage(v9, "region"));
}
if
(!regionsRentCancelled.isEmpty())
{
me.wiefferink.areashop.commands.plugin.message(sender, "add-rentCancelled",
me.wiefferink.areashop.tools.Utils.combinedMessage(regionsRentCancelled,
"region"));
}
if
(!v11.isEmpty()) {
me.wiefferink.areashop.commands.plugin.message(sender, "add-buyCancelled", me.wiefferink.areashop.tools.Utils.combinedMessage(v11,  "region"));
} if  (!v12.isEmpty()) {me.wiefferink.areashop.commands.plugin.message(sender, "add-blacklisted",
me.wiefferink.areashop.tools.Utils.createCommaSeparatedList(v12));

} if (!namesNoPermission.isEmpty()) {me.wiefferink.areashop.commands.plugin.message(sender, "add-noPermissionRegions", me.wiefferink.areashop.tools.Utils.createCommaSeparatedList(namesNoPermission));
me.wiefferink.areashop.commands.plugin.message(sender, "add-noPermissionOwnerMember");
}
if (!namesAddCancelled.isEmpty()) {
me.wiefferink.areashop.commands.plugin.message(sender, "add-rentCancelled",  me.wiefferink.areashop.tools.Utils.createCommaSeparatedList(namesAddCancelled));}});
}
