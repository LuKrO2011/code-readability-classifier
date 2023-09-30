	@Override
	public String getHelp(CommandSender target) {
		if(target.hasPermission("areashop.createrent")
				|| target.hasPermission("areashop.createrent.member")
				|| target.hasPermission("areashop.createrent.owner")

				|| target.hasPermission("areashop.createbuy")
				|| target.hasPermission("areashop.createbuy.member")
				|| target.hasPermission("areashop.createbuy.owner")) {
			return "help-add";
		}
		return null;
	}
