public java.util.List<java.lang.String>  getTabCompleteList(int
toComplete,
java.lang.String[]  start,
org.bukkit.command.CommandSender sender)
{

java.util.List<java.lang.String> result =
new java.util.ArrayList<>();
if  (toComplete
== 2)  {if (sender.hasPermission("areashop.createrent")) {result.add("rent");}
if (sender.hasPermission("areashop.createbuy"))
{
result.add("buy");
}
}
else if  (toComplete ==
3) {
if  (sender instanceof  org.bukkit.entity.Player) {
org.bukkit.entity.Player  v36 =  ((org.bukkit.entity.Player) (sender)); if (sender.hasPermission("areashop.createrent") ||  sender.hasPermission("areashop.createbuy"))  {
for (com.sk89q.worldguard.protection.regions.ProtectedRegion region
: plugin.getRegionManager(v36.getWorld()).getRegions().values()) {result.add(region.getId());

}
}

}

} return result; }
